import concurrent.futures
import functools
import multiprocessing
import os
import threading
import uuid
from collections.abc import Callable
from typing import Any, Self

import gevent.lock
import torch
from cyy_naive_lib.concurrency import BlockingSubmitExecutor, ProcessPoolWithCouroutine
from cyy_naive_lib.decorator import Decorator
from cyy_naive_lib.log import (
    log_debug,
    log_error,
    log_info,
    log_warning,
)
from cyy_naive_lib.storage import GlobalStore
from cyy_naive_lib.system_info import OSType, get_operating_system_type
from cyy_naive_lib.time_counter import TimeCounter
from cyy_naive_lib.topology import (
    CentralTopology,
    ClientEndpoint,
    ProcessPipeCentralTopology,
    ProcessQueueCentralTopology,
    ServerEndpoint,
)
from cyy_torch_toolbox.concurrency import TorchProcessContext
from cyy_torch_toolbox.device import (
    DeviceGreedyAllocator,
)
from cyy_torch_toolbox.device import (
    get_device_memory_info as _get_device_memory_info,
)

from .task_type import TaskIDType


class ThreadStore:
    __thread_data: None | threading.local = None

    @classmethod
    def __thread_local_data(cls) -> threading.local:
        if cls.__thread_data is None:
            cls.__thread_data = threading.local()
        return cls.__thread_data

    def store(self, name: str, obj: Any) -> None:
        setattr(self.__thread_local_data(), name, obj)

    def has(self, name: str) -> bool:
        if self.__thread_data is None:
            return False
        return hasattr(self.__thread_local_data(), name)

    def get(self, name: str) -> Any:
        if not self.has(name):
            return None
        return getattr(self.__thread_local_data(), name)


class ExecutorContext:
    __thread_store: None | ThreadStore = None
    coroutine_semaphore: None | gevent.lock.BoundedSemaphore = None

    def __init__(
        self,
        name: str | None = None,
    ) -> None:
        self.__name = name if name is not None else "unknown executor"
        self.__hold_device_lock: bool = False
        self.__used_device_memory = None
        self.global_store = GlobalStore(
            manager=TorchProcessContext().get_ctx().Manager()
        )
        if self.global_store.get_with_default("device_lock", None) is None:
            self.global_store.store_lock("device_lock")

    @property
    def thread_local_store(self) -> ThreadStore:
        if ExecutorContext.__thread_store is None:
            ExecutorContext.__thread_store = ThreadStore()
        assert ExecutorContext.__thread_store is not None
        return ExecutorContext.__thread_store

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            log_error("Found exception: %s %s %s", exc_type, exc, tb)
        self.release()

    def set_name(self, name: str) -> None:
        self.__name = name

    def acquire(self, cond_fun: Callable | None = None) -> None:
        if cond_fun is not None:
            while not cond_fun():
                gevent.sleep(0.1)
        if ExecutorContext.coroutine_semaphore is None:
            ExecutorContext.coroutine_semaphore = gevent.lock.BoundedSemaphore(value=1)
        ExecutorContext.coroutine_semaphore.acquire()
        self.__set_proc_name(self.__name)
        log_debug("get lock %s", self.coroutine_semaphore)

    @classmethod
    def __set_proc_name(cls, name: str) -> None:
        multiprocessing.current_process().name = name
        threading.current_thread().name = name

    def release(self) -> None:
        log_debug("release lock %s", self.coroutine_semaphore)
        self.release_device_lock()
        self.__set_proc_name("unknown executor")
        if ExecutorContext.coroutine_semaphore is not None:
            ExecutorContext.coroutine_semaphore.release()

    @functools.cached_property
    def device_lock(self) -> threading.RLock:
        return self.global_store.get("device_lock")

    def get_device(
        self, lock_callback: None | Callable = None, set_visible_device: bool = False
    ) -> torch.device:
        if not self.thread_local_store.has("device"):
            if not self.__hold_device_lock:
                self.device_lock.acquire()
                log_debug(
                    "lock device for process %s",
                    os.getpid(),
                )
                self.__hold_device_lock = True
                if lock_callback is not None:
                    lock_callback()
            device = DeviceGreedyAllocator.get_device(
                max_needed_bytes=self.__used_device_memory
            )
            self.thread_local_store.store("device", device)
            log_debug(
                "get device %s for process %s",
                device,
                os.getpid(),
            )
            if set_visible_device and "cuda" in device.type.lower():
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
        return self.thread_local_store.get("device")

    def release_device_lock(self) -> None:
        if self.__hold_device_lock:
            device: torch.device = self.thread_local_store.get("device")
            if "cuda" in device.type.lower():
                stats = torch.cuda.memory_stats(device=device)
                if "allocated_bytes.all.peak" in stats:
                    self.__used_device_memory = stats["allocated_bytes.all.peak"]
            self.device_lock.release()
            self.__hold_device_lock = False


class ClientEndpointInCoroutine(Decorator):
    def __init__(self, endpoint: ClientEndpoint, context: ExecutorContext) -> None:
        super().__init__(endpoint)
        self.__context = context

    def replace_context(self, context: ExecutorContext) -> None:
        self.__context = context

    def get(self) -> Any:
        self.__context.release()
        self.__context.acquire(cond_fun=self.has_data)
        return self._decorator_object.get()


class FederatedLearningContext(ExecutorContext):
    def __init__(self, worker_num: int, blocking_launch: bool) -> None:
        super().__init__()
        self.__worker_num = worker_num
        self.id = str(uuid.uuid4())
        self.blocking_launch = blocking_launch
        topology_class = ProcessPipeCentralTopology
        if get_operating_system_type() == OSType.Windows or "no_pipe" in os.environ:
            topology_class = ProcessQueueCentralTopology
            log_warning("use ProcessQueueCentralTopology")
        self.topology: CentralTopology = topology_class(
            mp_context=TorchProcessContext(), worker_num=self.__worker_num
        )
        self.__executor_pool: ProcessPoolWithCouroutine | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_FederatedLearningContext__executor_pool", None)
        return state

    def hold_semaphore(self, semaphore_name: str) -> bool:
        semaphore = self.global_store.get_semaphore(semaphore_name)
        return semaphore.acquire(blocking=False)

    def create_client_endpoint(
        self, endpoint_cls: type = ClientEndpoint, **endpoint_kwargs: Any
    ) -> ClientEndpointInCoroutine:
        return ClientEndpointInCoroutine(
            endpoint_cls(topology=self.topology, **endpoint_kwargs), context=self
        )

    def create_server_endpoint(
        self, endpoint_cls: type = ServerEndpoint, **endpoint_kwargs: Any
    ) -> ServerEndpoint:
        return endpoint_cls(topology=self.topology, **endpoint_kwargs)

    @property
    def executor_pool(self) -> ProcessPoolWithCouroutine:
        if self.__executor_pool is None:
            self.__executor_pool = ProcessPoolWithCouroutine(
                mp_context=TorchProcessContext().get_ctx(),
                initargs={
                    "process_data": {
                        "context": self,
                    }
                },
            )
            if self.blocking_launch:
                self.__executor_pool.wrap_executor(BlockingSubmitExecutor)
                assert isinstance(self.__executor_pool.executor, BlockingSubmitExecutor)
                self.__executor_pool.executor.set_global_store(self.global_store)
        return self.__executor_pool

    @property
    def submit_batch(self):
        return self.executor_pool.submit_batch

    @property
    def submit(self):
        return self.executor_pool.submit

    @property
    def wait_results(self):
        return self.executor_pool.wait_results

    def shutdown(self) -> None:
        self.executor_pool.shutdown()


class ConcurrentFederatedLearningContext:
    def __init__(self) -> None:
        self.__contexts: dict[TaskIDType, FederatedLearningContext] = {}
        self.context_info: dict[TaskIDType, dict] = {}
        self.__finished_tasks: set[TaskIDType] = set()

    def add_context(
        self, task_id: TaskIDType, context: FederatedLearningContext, **other_info
    ) -> None:
        assert task_id not in self.__contexts
        self.__contexts[task_id] = context
        self.context_info[task_id] = other_info

    def finished(self, task_id: TaskIDType) -> bool:
        return task_id in self.__finished_tasks

    def wait_results(
        self,
        timeout: float | None = None,
        return_when=concurrent.futures.ALL_COMPLETED,
    ) -> tuple[dict, int]:
        res: dict = {}
        remaining_jobs: int = 0
        timeout_ms: float | None = None
        if timeout is not None:
            timeout_ms = timeout * 1000
        for task_id, context in list(self.__contexts.items()):
            with TimeCounter() as counter:
                task_results, unfinised_cnt = context.wait_results(
                    timeout=timeout_ms / 1000 if timeout_ms is not None else None,
                    return_when=return_when,
                )
                if timeout_ms is not None:
                    timeout_ms = max(timeout_ms - counter.elapsed_milliseconds(), 0)
                remaining_jobs += unfinised_cnt
                if task_results:
                    res[task_id] = task_results
                if unfinised_cnt == 0:
                    context = self.__contexts.pop(task_id)
                    context.shutdown()
                    res[task_id] |= self.context_info.pop(task_id)
                    self.__finished_tasks.add(task_id)
        return res, remaining_jobs

    def release(self) -> dict:
        res, _ = self.wait_results()
        for context in self.__contexts.values():
            context.executor_pool.shutdown()
        self.__contexts.clear()
        return res


MB = 1024 * 1024
GB = MB * 1024


def get_device_memory_info(
    least_memory_GB: int | None = None,
) -> dict:
    memory_info = _get_device_memory_info()
    refined_memory_info: dict = {}
    log_info("before refine memory info %s", memory_info)
    if least_memory_GB is None:
        value = os.getenv("LEAST_REQUIRED_DEVICE_MEMORY_IN_GB")
        if value is not None:
            least_memory_GB = min(int(value), 1)
    for device, info in memory_info.items():
        if least_memory_GB is None:
            refined_memory_info[device] = info.free
            continue
        if info.free / GB < least_memory_GB:
            continue
        if info.used / info.total > 0.9:
            continue
        free_GB = int(info.free / GB)
        if free_GB == 0:
            continue
        refined_memory_info[device] = info.free
    assert refined_memory_info, "No avaiable device"
    log_info("refined memory info %s", refined_memory_info)
    return refined_memory_info


def allocate_device(
    worker_number: int,
    count_server: bool,
    least_memory_GB: int | None = None,
) -> dict:
    refined_memory_info = get_device_memory_info(least_memory_GB=least_memory_GB)
    refined_memory_info_list = sorted(
        list(refined_memory_info.items()), key=lambda a: a[1], reverse=True
    )
    free_bytes = list(a[1] for a in refined_memory_info_list)
    devices = list(a[0] for a in refined_memory_info_list)
    devices = devices * 2
    log_warning("Use devices %s", devices)
    if count_server:
        result = {"server_device": devices[0]}
        devices = devices[0:]
        free_bytes = free_bytes[1:]
        if not free_bytes:
            result |= {"worker_number_per_process": worker_number}
            result |= {"process_devices": devices}
            return result
    else:
        result = {"server_device": devices[-1]}
    if worker_number <= len(free_bytes):
        result |= {"worker_number_per_process": 1}
        result |= {"process_devices": devices}
        return result
    # small scale training
    if worker_number <= 50:
        worker_number_per_process = max(int(worker_number / len(free_bytes) - 1), 1)
        while worker_number / worker_number_per_process > len(free_bytes):
            worker_number_per_process += 1
        result |= {"worker_number_per_process": worker_number_per_process}
        result |= {"process_devices": devices}
        return result
    total_bytes = sum(free_bytes)
    MB_per_worker = min(total_bytes / MB / worker_number, 10 * GB)
    log_debug(
        "MB_per_worker %s other %s",
        MB_per_worker,
        min(free_bytes) / MB,
    )
    worker_number_per_process = max(int(min(free_bytes) / MB / MB_per_worker), 1)
    result |= {"worker_number_per_process": worker_number_per_process}
    result |= {"process_devices": devices}
    return result

import concurrent.futures
import functools
import multiprocessing
import os
import threading
from collections.abc import Callable
from typing import Any, Self

import gevent.lock
import torch
from cyy_naive_lib.decorator import Decorator
from cyy_naive_lib.log import (
    log_debug,
    log_error,
    log_info,
    log_warning,
)
from cyy_naive_lib.system_info import OSType, get_operating_system_type
from cyy_naive_lib.time_counter import TimeCounter
from cyy_naive_lib.topology import (
    CentralTopology,
    ClientEndpoint,
    ProcessPipeCentralTopology,
    ProcessQueueCentralTopology,
    ServerEndpoint,
)
from cyy_torch_toolbox import get_device
from cyy_torch_toolbox.concurrency import TorchProcessContext
from cyy_torch_toolbox.device import get_device_memory_info, set_device

from .concurrency import CoroutineExcutorPool
from .task import TaskIDType


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


class GlobalStore:
    global_manager: None | Any = None
    objects: None | dict = None

    def __init__(self) -> None:
        if GlobalStore.global_manager is None:
            GlobalStore.global_manager = TorchProcessContext().get_ctx().Manager()
        if GlobalStore.objects is None:
            assert GlobalStore.global_manager is not None
            GlobalStore.objects = GlobalStore.global_manager.dict()
            self.store_lock("default")

    def store_lock(self, name: str) -> None:
        assert self.global_manager is not None
        self.store(name, self.global_manager.RLock())

    def store(self, name: str, obj: Any) -> None:
        assert self.objects is not None
        assert name not in self.objects
        self.objects[name] = obj

    def set_default_with_manager_object(self, name: str, type_name: str) -> Any:
        assert self.objects is not None
        self.objects.setdefault(name, getattr(self.global_manager, type_name)())

    def get_with_default(self, name: str, default: Any = None) -> Any:
        assert self.objects is not None
        return self.objects.get(name, default)

    def get(self, name: str) -> Any:
        assert self.objects is not None
        return self.objects[name]


class ExecutorContext:
    __global_store: None | GlobalStore = None
    __thread_store: None | ThreadStore = None
    coroutine_semaphore: None | gevent.lock.BoundedSemaphore = None

    def __init__(
        self,
        name: str | None = None,
    ) -> None:
        self.__name = name if name is not None else "unknown executor"
        self.__hold_device_lock: bool = False
        self.__used_device_memory = None
        self.global_store.store_lock("device_lock")

    @property
    def global_store(self) -> GlobalStore:
        if ExecutorContext.__global_store is None:
            ExecutorContext.__global_store = GlobalStore()
        assert ExecutorContext.__global_store is not None
        return ExecutorContext.__global_store

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

    @property
    def device_lock(self) -> threading.RLock:
        return self.global_store.get("device_lock")

    def get_device(self, lock_callback: None | Callable = None) -> torch.device:
        if not self.thread_local_store.has("device"):
            if not self.__hold_device_lock:
                self.device_lock.acquire()
                self.__hold_device_lock = True
                if lock_callback is not None:
                    lock_callback()
            device = get_device(max_needed_bytes=self.__used_device_memory)
            self.thread_local_store.store("device", device)
            log_debug(
                "get device %s for process %s",
                device,
                os.getpid(),
            )
            set_device(device)
        return self.thread_local_store.get("device")

    def release_device_lock(self) -> None:
        if self.__hold_device_lock:
            device: torch.device = self.thread_local_store.get("device")
            if "cuda" in device.type.lower():
                stats = torch.cuda.memory_stats(device=device)
                if "allocated_bytes.all.peak" in stats:
                    self.__used_device_memory = stats["allocated_bytes.all.peak"]
            log_info("release device_lock ")
            assert self.device_lock is not None
            self.device_lock.release()
            self.__hold_device_lock = False


class ClientEndpointInCoroutine(Decorator):
    def __init__(self, endpoint: ClientEndpoint, context: ExecutorContext) -> None:
        super().__init__(endpoint)
        self.__context = context

    def get(self) -> Any:
        self.__context.release()
        self.__context.acquire(cond_fun=self.has_data)
        return self._decorator_object.get()


class FederatedLearningContext(ExecutorContext):
    def __init__(self, worker_num: int) -> None:
        super().__init__()
        self.__worker_num = worker_num
        topology_class = ProcessPipeCentralTopology
        if get_operating_system_type() == OSType.Windows or "no_pipe" in os.environ:
            topology_class = ProcessQueueCentralTopology
            log_warning("use ProcessQueueCentralTopology")
        self.topology: CentralTopology = topology_class(
            mp_context=TorchProcessContext(), worker_num=self.__worker_num
        )
        self.__executor_pool: CoroutineExcutorPool | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_FederatedLearningContext__executor_pool", None)
        return state

    def hold_semaphore(self, semaphore_name: str) -> bool:
        semaphore = self.global_store.get_with_default(semaphore_name, None)
        if semaphore is None:
            semaphore = self.global_store.set_default_with_manager_object(
                semaphore_name, "Semaphore"
            )
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
    def executor_pool(self) -> CoroutineExcutorPool:
        if self.__executor_pool is None:
            self.__executor_pool = CoroutineExcutorPool(
                initargs={
                    "process_data": {
                        "context": self,
                    }
                },
                pass_process_data=True,
            )
            self.__executor_pool.catch_exception()
        return self.__executor_pool

    @property
    def submit(self):
        return self.executor_pool.submit

    @property
    def wait_results(self):
        return self.executor_pool.wait_results

    def submit_batch(self, batch_fun: Callable, kwargs_list: list):
        return self.executor_pool.submit_batch(
            [
                functools.partial(batch_fun, **kwargs_elem)
                for kwargs_elem in kwargs_list
            ],
        )

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


def get_worker_number_per_process(
    worker_number: int, count_server: bool = False
) -> int:
    memory_info = get_device_memory_info()
    refined_memory_info: dict = {}
    MB = 1024 * 1024
    GB = MB * 1024
    for device, info in memory_info.items():
        if info.total / GB >= 20 and info.free / GB < 5:
            continue
        if info.used / info.total > 0.9:
            continue
        free_GB = int(info.free / GB)
        if free_GB == 0:
            continue
        refined_memory_info[device] = info.free
    assert refined_memory_info
    log_warning("Use devices %s", list(refined_memory_info.keys()))
    free_bytes = sorted(list(refined_memory_info.values()))
    if count_server:
        free_bytes = free_bytes[1:]
        if not free_bytes:
            return worker_number
    if worker_number <= len(free_bytes):
        return 1
    # small scale training
    if worker_number <= 50:
        res = max(int(worker_number / len(free_bytes) - 1), 1)
        while worker_number / res > len(free_bytes):
            res += 1
        return res
    total_bytes = sum(free_bytes)
    MB_per_worker = min(total_bytes / MB / worker_number, 10 * GB)
    log_debug(
        "MB_per_worker %s other %s",
        MB_per_worker,
        min(free_bytes) / MB,
    )
    worker_number_per_process = max(int(min(free_bytes) / MB / MB_per_worker), 1)
    return worker_number_per_process

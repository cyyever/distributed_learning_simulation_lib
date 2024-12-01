import concurrent.futures
import functools
import multiprocessing
import os
import threading
from collections.abc import Callable, Sequence
from typing import Any, Self

import gevent.lock
import torch
from cyy_naive_lib.concurrency.process_initialization import get_process_data
from cyy_naive_lib.log import (
    log_debug,
    log_error,
    log_warning,
)
from cyy_naive_lib.system_info import OSType, get_operating_system_type
from cyy_naive_lib.topology import (
    CentralTopology,
    ClientEndpoint,
    ProcessPipeCentralTopology,
    ProcessQueueCentralTopology,
    ServerEndpoint,
)
from cyy_torch_toolbox import get_device
from cyy_torch_toolbox.concurrency import TorchProcessContext, TorchProcessPool


class ExecutorContext:
    __thread_data: None | threading.local = None
    semaphore: None | gevent.lock.BoundedSemaphore = None
    # gevent.lock.BoundedSemaphore(value=1)

    def __init__(
        self,
        device_lock: threading.RLock,
        name: str | None = None,
    ) -> None:
        self.__name = name if name is not None else "unknown executor"
        self.__device_lock: threading.RLock = device_lock
        self.__hold_device_lock: bool = False
        self.__used_device_memory = None

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
        if ExecutorContext.semaphore is None:
            ExecutorContext.semaphore = gevent.lock.BoundedSemaphore(value=1)
        ExecutorContext.semaphore.acquire()
        multiprocessing.current_process().name = self.__name
        threading.current_thread().name = self.__name
        log_debug("get lock %s", self.semaphore)

    def release(self) -> None:
        log_debug("release lock %s", self.semaphore)
        self.release_device_lock()
        self.set_name("unknown executor")
        if ExecutorContext.semaphore is not None:
            ExecutorContext.semaphore.release()

    def get_device(self, lock_callback: None | Callable = None) -> torch.device:
        if self.__thread_data is None:
            self.__thread_data = threading.local()
        if not hasattr(self.__thread_data, "device"):
            if not self.__hold_device_lock:
                self.__device_lock.acquire()
                self.__hold_device_lock = True
                if lock_callback is not None:
                    lock_callback()
            self.__thread_data.device = get_device(
                max_needed_bytes=self.__used_device_memory
            )
            if "cuda" in self.__thread_data.device.type.lower():
                torch.cuda.set_device(self.__thread_data.device)
        return self.__thread_data.device

    def release_device_lock(self, **kwargs: Any) -> None:
        if self.__hold_device_lock:
            assert self.__thread_data is not None
            if "cuda" in self.__thread_data.device.type.lower():
                stats = torch.cuda.memory_stats(device=self.__thread_data.device)
                if stats:
                    self.__used_device_memory = stats["allocated_bytes.all.peak"]
            self.__device_lock.release()
            self.__hold_device_lock = False


class CoroutineExcutorPool(TorchProcessPool):
    def submit_batch(self, funs: Sequence[Callable]) -> concurrent.futures.Future:
        return super().submit(CoroutineExcutorPool.__batch_fun, funs)

    @classmethod
    def __batch_fun(cls, funs) -> None:
        assert funs
        gevent.joinall([gevent.spawn(fun) for fun in funs], raise_error=True)


def wrap_fun(fn: Callable, *args: Any, **kwargs: Any):
    context = get_process_data()["context"]
    return fn(*args, **kwargs, context=context)


class FederatedLearningContext(ExecutorContext):
    manager = multiprocessing.Manager()
    semaphores: dict = {}

    def __init__(self, worker_num: int) -> None:
        super().__init__(self.manager.RLock())
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
        # log_error("keys are %s", state.keys())
        return state

    @classmethod
    def create_semaphore(cls, semaphore_name: str) -> None:
        assert semaphore_name not in cls.semaphores
        cls.semaphores[semaphore_name] = cls.manager.Semaphore()

    @classmethod
    def get_semaphore(cls, semaphore_name: str) -> threading.Semaphore | None:
        return cls.semaphores.get(semaphore_name, None)

    def create_client_endpoint(
        self, end_point_cls: type, **endpoint_kwargs
    ) -> ClientEndpoint:
        return end_point_cls(topology=self.topology, **endpoint_kwargs)

    def create_server_endpoint(
        self, end_point_cls: type, **endpoint_kwargs
    ) -> ServerEndpoint:
        return end_point_cls(topology=self.topology, **endpoint_kwargs)

    @property
    def executor_pool(self) -> CoroutineExcutorPool:
        if self.__executor_pool is None:
            self.__executor_pool = CoroutineExcutorPool(
                initargs={
                    "process_data": {
                        "context": self,
                    }
                }
            )
            self.__executor_pool.catch_exception()
        return self.__executor_pool

    def submit(self, funs: Sequence[Callable], **kwargs: Any):
        log_error("keys %s", kwargs)
        assert funs
        if len(funs) == 1:
            if "kwargs_list" in kwargs:
                kwargs_list = kwargs.pop("kwargs_list")
                assert not kwargs
                assert len(kwargs_list) == 1
                kwargs = kwargs_list[0]

            return self.executor_pool.submit(
                functools.partial(
                    wrap_fun,
                    fn=funs[0],
                ),
                **kwargs,
            )

        assert len(kwargs) == 1
        return self.executor_pool.submit_batch(
            [
                functools.partial(wrap_fun, fn=fn, **kwargs_elem)
                for fn, kwargs_elem in zip(funs, kwargs["kwargs_list"])
            ],
        )

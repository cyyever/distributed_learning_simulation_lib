import multiprocessing
import os
import threading
from collections.abc import Callable
from typing import Any, Self

import gevent.lock
import torch
from cyy_naive_lib.log import log_debug, log_error, log_warning
from cyy_naive_lib.system_info import OSType, get_operating_system_type
from cyy_naive_lib.topology import (
    CentralTopology,
    ClientEndpoint,
    ProcessPipeCentralTopology,
    ProcessQueueCentralTopology,
    ServerEndpoint,
)
from cyy_torch_toolbox import get_device
from cyy_torch_toolbox.concurrency import TorchProcessContext


class ExecutorContext:
    __thread_data = threading.local()
    semaphore = gevent.lock.BoundedSemaphore(value=1)

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

    def acquire(self) -> None:
        self.semaphore.acquire()
        multiprocessing.current_process().name = self.__name
        threading.current_thread().name = self.__name
        log_debug("get lock %s", self.semaphore)

    def release(self) -> None:
        log_debug("release lock %s", self.semaphore)
        self.release_device_lock()
        self.set_name("unknown executor")
        self.semaphore.release()

    def get_device(self, lock_callback: None | Callable = None) -> torch.device:
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
            if "cuda" in self.__thread_data.device.type.lower():
                stats = torch.cuda.memory_stats(device=self.__thread_data.device)
                if stats:
                    self.__used_device_memory = stats["allocated_bytes.all.peak"]
            self.__device_lock.release()
            self.__hold_device_lock = False


class DeviceContext(ExecutorContext):
    pass


class FederatedLearningContext:
    manager = multiprocessing.Manager()

    def __init__(self, worker_num: int) -> None:
        self.__worker_num = worker_num
        topology_class = ProcessPipeCentralTopology
        if get_operating_system_type() == OSType.Windows or "no_pipe" in os.environ:
            topology_class = ProcessQueueCentralTopology
            log_warning("use ProcessQueueCentralTopology")
        self.topology: CentralTopology = topology_class(
            mp_context=TorchProcessContext(), worker_num=self.__worker_num
        )
        self.__executor_context = ExecutorContext(self.manager.RLock())

    def create_client_endpoint(
        self, end_point_cls: type, **endpoint_kwargs
    ) -> ClientEndpoint:
        return end_point_cls(topology=self.topology, **endpoint_kwargs)

    def create_server_endpoint(
        self, end_point_cls: type, **endpoint_kwargs
    ) -> ServerEndpoint:
        return end_point_cls(topology=self.topology, **endpoint_kwargs)

    def __enter__(self) -> Self:
        self.__executor_context.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.__executor_context.__exit__(exc_type, exc, tb)

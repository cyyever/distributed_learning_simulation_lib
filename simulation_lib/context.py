import multiprocessing
import threading
from collections.abc import Callable
from typing import Any, Self

import gevent.lock
import torch
from cyy_naive_lib.log import log_debug, log_error
from cyy_torch_toolbox import get_device


class DeviceContext:
    __thread_data = threading.local()
    semaphore = gevent.lock.BoundedSemaphore(value=1)

    def __init__(
        self,
        name: str,
        device_lock: threading.RLock,
    ) -> None:
        self.__name = name
        self.__device_lock: threading.RLock = device_lock
        self.__hold_device_lock: bool = False
        self.__used_device_memory = None

    def __enter__(self) -> Self:
        self.acquire(name=self.__name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            log_error("Found exception: %s %s %s", exc_type, exc, tb)
        self.release()

    @classmethod
    def set_name(cls, name: str) -> None:
        multiprocessing.current_process().name = name
        threading.current_thread().name = name

    def acquire(self, name: str) -> None:
        self.semaphore.acquire()
        self.set_name(name)
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

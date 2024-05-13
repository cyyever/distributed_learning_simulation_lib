import asyncio
import copy
import multiprocessing
import os
import threading
from functools import cached_property
from typing import Any, Callable, Self

import torch
from cyy_naive_lib.log import log_debug
from cyy_torch_toolbox.device import get_device

from .config import DistributedTrainingConfig


class ExecutorContext:
    semaphore = asyncio.BoundedSemaphore(value=1)

    def __init__(self, name: str) -> None:
        self.__name = name

    async def __aenter__(self) -> Self:
        await self.acquire(name=self.__name)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()

    @classmethod
    async def acquire(cls, name: str) -> None:
        await cls.semaphore.acquire()
        multiprocessing.current_process().name = name
        threading.current_thread().name = name
        log_debug("get lock %s", cls.semaphore)

    @classmethod
    def release(cls) -> None:
        log_debug("release lock %s", cls.semaphore)
        multiprocessing.current_process().name = "unknown executor"
        threading.current_thread().name = "unknown executor"
        cls.semaphore.release()


class Executor:
    __thread_data = threading.local()

    def __init__(
        self,
        config: DistributedTrainingConfig,
        name: str,
        device_lock: threading.RLock,
        log_lock: threading.Semaphore | None = None,
    ) -> None:
        self.__config: DistributedTrainingConfig = copy.deepcopy(config)
        self.__used_device_memory = None
        self.__name = name
        self.__device_lock: threading.RLock = device_lock
        self.__hold_device_lock: bool = False
        self.__log_lock: threading.Semaphore | None = log_lock
        self.__hold_log_lock: bool | None = None

    @property
    def name(self) -> str:
        return self.__name

    @property
    def hold_log_lock(self) -> bool:
        if self.__hold_log_lock is not None:
            return self.__hold_log_lock
        if self.__log_lock is None:
            self.__hold_log_lock = False
            return self.__hold_log_lock
        self.__hold_log_lock = self.__log_lock.acquire(blocking=False)
        return self.__hold_log_lock

    @property
    def config(self) -> DistributedTrainingConfig:
        return self.__config

    @cached_property
    def save_dir(self) -> str:
        assert self.config.get_save_dir()
        executor_save_dir = os.path.abspath(
            os.path.join(self.config.get_save_dir(), self.name.replace(" ", "_"))
        )
        os.makedirs(executor_save_dir, exist_ok=True)
        return executor_save_dir

    def _get_device(self, lock_callback: None | Callable = None) -> torch.device:
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

    async def start(self) -> None:
        raise NotImplementedError()

import copy
import os
import threading
from functools import cached_property

from .config import DistributedTrainingConfig
from .context import DeviceContext


class Executor:
    def __init__(
        self,
        config: DistributedTrainingConfig,
        name: str,
        device_lock: threading.RLock,
        log_lock: threading.Semaphore | None = None,
    ) -> None:
        self.__config: DistributedTrainingConfig = copy.deepcopy(config)
        self.__name = name
        self.__log_lock: threading.Semaphore | None = log_lock
        self.__hold_log_lock: bool | None = None
        self.__device_context = DeviceContext(name=self.name, device_lock=device_lock)

    @property
    def device_context(self) -> DeviceContext:
        return self.__device_context

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

    def start(self) -> None:
        raise NotImplementedError()

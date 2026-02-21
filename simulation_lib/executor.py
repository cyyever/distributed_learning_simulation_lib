import copy
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

from .config import DistributedTrainingConfig
from .context import FederatedLearningContext
from .task_type import TaskIDType


class Executor(ABC):
    def __init__(
        self,
        config: DistributedTrainingConfig,
        context: FederatedLearningContext,
        name: str,
        task_id: TaskIDType,
    ) -> None:
        self.__config: DistributedTrainingConfig = copy.deepcopy(config)
        self.__context: FederatedLearningContext = copy.copy(context)
        self.__context.set_name(name)
        self.__name = name
        self.__task_id = task_id
        self.__has_log_lock: bool | None = None

    @property
    def hold_log_lock(self) -> bool:
        if self.__has_log_lock is not None:
            return self.__has_log_lock
        self.__has_log_lock = self.context.hold_semaphore("log_lock")
        return self.__has_log_lock

    @property
    def context(self) -> FederatedLearningContext:
        return self.__context

    @property
    def task_id(self) -> TaskIDType:
        return self.__task_id

    @property
    def name(self) -> str:
        return self.__name

    @property
    def config(self) -> DistributedTrainingConfig:
        return self.__config

    @cached_property
    def save_dir(self) -> Path:
        assert self.config.get_save_dir()
        executor_save_dir = (
            Path(self.config.get_save_dir()) / self.name.replace(" ", "_")
        ).resolve()
        executor_save_dir.mkdir(parents=True, exist_ok=True)
        return executor_save_dir

    @abstractmethod
    def start(self) -> None: ...

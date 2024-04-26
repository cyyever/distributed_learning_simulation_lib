from typing import Protocol

from .config import DistributedTrainingConfig


class ExecutorProtocol(Protocol):
    @property
    def hold_log_lock(self) -> bool:
        ...

    @property
    def config(self) -> DistributedTrainingConfig:
        ...

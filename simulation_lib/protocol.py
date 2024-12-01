from typing import Protocol

from .config import DistributedTrainingConfig
from .context import FederatedLearningContext


class ExecutorProtocol(Protocol):
    @property
    def context(self) -> FederatedLearningContext: ...
    @property
    def hold_log_lock(self) -> bool: ...

    @property
    def config(self) -> DistributedTrainingConfig: ...

    @property
    def name(self) -> str: ...

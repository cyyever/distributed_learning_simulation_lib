from typing import Protocol

from .config import DistributedTrainingConfig
from .context import DeviceContext


class ExecutorProtocol(Protocol):
    @property
    def device_context(self) -> DeviceContext: ...
    @property
    def hold_log_lock(self) -> bool: ...

    @property
    def config(self) -> DistributedTrainingConfig: ...

    @property
    def name(self) -> str: ...

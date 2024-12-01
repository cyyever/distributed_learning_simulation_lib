from typing import Protocol

from cyy_naive_lib.topology import Endpoint
from cyy_torch_toolbox import Trainer

from ..protocol import ExecutorProtocol
from ..util import ModelCache


class TrainerProtocol(Protocol):
    @property
    def trainer(self) -> Trainer: ...


class WorkerProtocol(ExecutorProtocol):
    @property
    def endpoint(self) -> Endpoint: ...

    @property
    def trainer(self) -> Trainer: ...

    def pause(self, in_round: bool) -> None: ...


class AggregationWorkerProtocol(WorkerProtocol):
    @property
    def round_index(self) -> int: ...

    @property
    def model_cache(self) -> ModelCache: ...

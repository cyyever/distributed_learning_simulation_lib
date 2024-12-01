from typing import Protocol

from cyy_naive_lib.topology import Endpoint

from ..protocol import ExecutorProtocol
from ..util import ModelCache


class TrainerProtocol(Protocol):
    pass


class WorkerProtocol(ExecutorProtocol):
    @property
    def endpoint(self) -> Endpoint: ...

    def pause(self, in_round: bool) -> None: ...


class AggregationWorkerProtocol(WorkerProtocol):
    @property
    def round_index(self) -> int: ...

    @property
    def model_cache(self) -> ModelCache: ...

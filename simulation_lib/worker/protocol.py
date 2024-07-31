from functools import cached_property

from cyy_naive_lib.topology import Endpoint
from cyy_torch_toolbox import Trainer

from ..protocol import ExecutorProtocol
from ..util import ModelCache


class WorkerProtocol(ExecutorProtocol):

    @property
    def endpoint(self) -> Endpoint: ...

    @cached_property
    def trainer(self) -> Trainer: ...

    def pause(self) -> None: ...


class AggregationWorkerProtocol(WorkerProtocol):

    @property
    def round_index(self) -> int: ...

    @property
    def model_cache(self) -> ModelCache: ...

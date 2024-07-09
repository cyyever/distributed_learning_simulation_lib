from cyy_torch_toolbox import Trainer
from cyy_naive_lib.topology import Endpoint
from functools import cached_property
from ..util import ModelCache
from ..protocol import ExecutorProtocol


class WorkerProtocol(ExecutorProtocol):

    @property
    def endpoint(self) -> Endpoint:
        ...

    @cached_property
    def trainer(self) -> Trainer:
        ...

    def pause(self) -> None:
        ...


class AggregationWorkerProtocol(WorkerProtocol):

    @property
    def round_index(self) -> int:
        ...

    @property
    def model_cache(self) -> ModelCache:
        ...


class GraphWorkerProtocol(AggregationWorkerProtocol):

    @cached_property
    def training_node_indices(self) -> set:
        ...

from typing import Iterable

from cyy_torch_toolbox import Trainer
from cyy_naive_lib.topology.endpoint import Endpoint

from ..util import ModelCache
from ..protocol import ExecutorProtocol


class WorkerProtocol(ExecutorProtocol):

    @property
    def endpoint(self) -> Endpoint:
        ...

    @property
    def trainer(self) -> Trainer:
        ...

class AggregationWorkerProtocol(WorkerProtocol):


    @property
    def round_index(self) -> int:
        ...

    @property
    def model_cache(self) -> ModelCache:
        ...


class GraphWorkerProtocol(AggregationWorkerProtocol):
    @property
    def training_node_indices(self) -> Iterable[int]:
        ...

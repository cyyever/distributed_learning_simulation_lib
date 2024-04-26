from typing import Iterable

from cyy_torch_toolbox import Trainer

from ..util import ModelCache
from ..protocol import ExecutorProtocol


class AggregationWorkerProtocol(ExecutorProtocol):

    @property
    def trainer(self) -> Trainer:
        ...

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

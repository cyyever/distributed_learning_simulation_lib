from typing import Iterable, Protocol

from cyy_torch_toolbox import Trainer

from ..config import DistributedTrainingConfig
from ..util import ModelCache


class AggregationWorkerProtocol(Protocol):
    @property
    def trainer(self) -> Trainer:
        ...

    @property
    def round_index(self) -> int:
        ...

    @property
    def model_cache(self) -> ModelCache:
        ...

    @property
    def config(self) -> DistributedTrainingConfig:
        ...


class GraphWorkerProtocol(AggregationWorkerProtocol):
    @property
    def training_node_indices(self) -> Iterable[int]:
        ...

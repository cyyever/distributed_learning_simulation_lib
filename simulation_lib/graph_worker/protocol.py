from functools import cached_property
from typing import Protocol

from ..worker.protocol import AggregationWorkerProtocol


class GraphWorkerProtocol(AggregationWorkerProtocol, Protocol):
    @cached_property
    def training_node_indices(self) -> set: ...

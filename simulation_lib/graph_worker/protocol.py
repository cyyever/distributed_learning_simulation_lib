from functools import cached_property

from ..worker.protocol import AggregationWorkerProtocol


class GraphWorkerProtocol(AggregationWorkerProtocol):

    @cached_property
    def training_node_indices(self) -> set: ...

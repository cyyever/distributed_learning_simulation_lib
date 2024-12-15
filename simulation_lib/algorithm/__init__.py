from .aggregation_algorithm import AggregationAlgorithm
from .composite_aggregation_algorithm import CompositeAggregationAlgorithm
from .fed_avg_algorithm import FedAVGAlgorithm
from .personalized_aggregation_algorithm import PersonalizedFedAVGAlgorithm

__all__ = [
    "AggregationAlgorithm",
    "CompositeAggregationAlgorithm",
    "FedAVGAlgorithm",
    "PersonalizedFedAVGAlgorithm",
]

try:
    from .graph_algorithm import GraphAlgorithm
    from .graph_embedding_algorithm import GraphNodeEmbeddingPassingAlgorithm
    from .graph_topology_algorithm import GraphTopologyAlgorithm

    __all__ += [
        "GraphAlgorithm",
        "GraphNodeEmbeddingPassingAlgorithm",
        "GraphTopologyAlgorithm",
    ]
except Exception:
    pass

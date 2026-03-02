from cyy_torch_toolbox import TorchProcessContext

from .algorithm import (
    AggregationAlgorithm,
    CompositeAggregationAlgorithm,
    FedAVGAlgorithm,
    PersonalizedFedAVGAlgorithm,
)
from .algorithm_repository import AlgorithmRepository
from .analysis import *  # noqa: F401
from .common_method import *  # noqa: F401
from .config import DistributedTrainingConfig, load_config
from .evaluation import get_server
from .message import (
    DeltaParameterMessage,
    FeatureMessage,
    Message,
    MultipleWorkerMessage,
    ParameterMessage,
    ParameterMessageBase,
)
from .protocol import ExecutorProtocol
from .server import AggregationServer
from .session import Session
from .topology import (
    DifferentialPrivacyEmbeddingEndpoint,
    NNADQClientEndpoint,
    NNADQServerEndpoint,
    QuantClientEndpoint,
    QuantServerEndpoint,
    StochasticQuantClientEndpoint,
    StochasticQuantServerEndpoint,
)
from .training import train
from .util import ModelCache
from .worker import (
    AggregationWorker,
    AggregationWorkerProtocol,
    ErrorFeedbackWorker,
    GradientWorker,
    Worker,
)

__all__ = [
    "AlgorithmRepository",
    "ExecutorProtocol",
    "get_server",
    "train",
    "Session",
    "DifferentialPrivacyEmbeddingEndpoint",
    "QuantServerEndpoint",
    "QuantClientEndpoint",
    "StochasticQuantClientEndpoint",
    "StochasticQuantServerEndpoint",
    "NNADQClientEndpoint",
    "NNADQServerEndpoint",
    "Message",
    "ParameterMessageBase",
    "ParameterMessage",
    "DeltaParameterMessage",
    "FeatureMessage",
    "MultipleWorkerMessage",
    "AggregationWorker",
    "ErrorFeedbackWorker",
    "GradientWorker",
    "Worker",
    "ModelCache",
    "AggregationWorkerProtocol",
    "AggregationServer",
    "AggregationAlgorithm",
    "CompositeAggregationAlgorithm",
    "FedAVGAlgorithm",
    "PersonalizedFedAVGAlgorithm",
    "load_config",
    "DistributedTrainingConfig",
]
try:
    from .algorithm import (
        GraphAlgorithm,
        GraphNodeEmbeddingPassingAlgorithm,
        GraphTopologyAlgorithm,
    )

    __all__ += [
        "GraphAlgorithm",
        "GraphNodeEmbeddingPassingAlgorithm",
        "GraphTopologyAlgorithm",
    ]
except ImportError:
    pass

from ..algorithm import FedAVGAlgorithm
from ..algorithm_repository import AlgorithmRepository
from ..server import AggregationServer
from ..worker import AggregationWorker

AlgorithmRepository.register_algorithm(
    algorithm_name="fed_avg",
    client_cls=AggregationWorker,
    server_cls=AggregationServer,
    algorithm_cls=FedAVGAlgorithm,
)

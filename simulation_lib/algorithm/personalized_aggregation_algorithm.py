import copy

from ..message import Message, MultipleWorkerMessage, ParameterMessage
from .aggregation_algorithm import AggregationAlgorithm
from .fed_avg_algorithm import FedAVGAlgorithm


class PersonalizedFedAVGAlgorithm(AggregationAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self._worker_aggregation_algorithms: dict[int, FedAVGAlgorithm] = {}
        self._worker_weights: dict[int, dict[int, float]] = {}

    def set_worker_weights(self, worker_weights: dict[int, dict[int, float]]) -> None:
        assert not self._worker_weights
        assert not self._worker_aggregation_algorithms
        self._worker_weights = worker_weights
        self._worker_aggregation_algorithms = {
            worker_id: FedAVGAlgorithm() for worker_id in self._worker_weights
        }

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: Message | None,
    ) -> bool:
        assert self._worker_weights
        assert self._worker_aggregation_algorithms
        for other_worker_id in self._worker_weights:
            if other_worker_id == worker_id:
                continue
            weight = self._worker_weights[other_worker_id].get(worker_id, 1000000)
            worker_data_copy = worker_data
            if worker_data_copy is not None:
                worker_data_copy = copy.deepcopy(worker_data)
                assert isinstance(worker_data_copy, ParameterMessage)
                worker_data_copy.aggregation_weight = weight
            self._worker_aggregation_algorithms[other_worker_id].process_worker_data(
                worker_id=worker_id, worker_data=worker_data_copy
            )
        return True

    def aggregate_worker_data(self) -> Message:
        worker_data = {
            worker_id: algorithm.aggregate_worker_data()
            for worker_id, algorithm in self._worker_aggregation_algorithms.items()
        }
        centralized_parameter_dict = AggregationAlgorithm.weighted_avg(
            worker_data, 1 / len(worker_data)
        )
        return MultipleWorkerMessage(
            worker_data=worker_data,
            other_data={"centralized_parameter": centralized_parameter_dict},
        )

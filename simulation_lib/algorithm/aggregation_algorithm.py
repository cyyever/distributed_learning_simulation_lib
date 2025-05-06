from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
from cyy_torch_toolbox import ModelParameter

from ..config import DistributedTrainingConfig
from ..message import Message, ParameterMessage


class AggregationAlgorithm:
    def __init__(self) -> None:
        self._all_worker_data: MutableMapping[int, Message] = {}
        self.__skipped_workers: set[int] = set()
        self._old_parameter: ModelParameter | None = None
        self._config: DistributedTrainingConfig | None = None

    def set_old_parameter(self, old_parameter: ModelParameter) -> None:
        self._old_parameter = old_parameter

    @property
    def config(self) -> DistributedTrainingConfig:
        assert self._config is not None
        return self._config

    def set_config(self, config: DistributedTrainingConfig) -> None:
        self._config = config

    @classmethod
    def get_total_weight(cls, data_dict: Mapping[int, Message]) -> float:
        total_weight: float = 0
        for v in data_dict.values():
            assert v.aggregation_weight is not None
            total_weight += v.aggregation_weight
        return total_weight

    @classmethod
    def get_ratios(cls, data_dict: Mapping[int, Message]) -> dict[int, float]:
        total_weight: float = float(cls.get_total_weight(data_dict=data_dict))
        ratios = {}
        for k, v in data_dict.items():
            assert v.aggregation_weight is not None
            ratios[k] = float(v.aggregation_weight) / total_weight
        return ratios

    @classmethod
    def weighted_avg(
        cls,
        data_dict: Mapping[int, Message],
        weights: dict[int, float] | float,
    ) -> ModelParameter:
        assert data_dict
        avg_data: ModelParameter = {}
        for worker_id, v in data_dict.items():
            weight = weights[worker_id] if isinstance(weights, dict) else weights
            assert 0 <= weight <= 1
            assert isinstance(v, ParameterMessage)
            d = {
                k2: v2.to(dtype=torch.float64) * weight
                for (k2, v2) in v.parameter.items()
            }
            assert d
            if not avg_data:
                avg_data = d
            else:
                for k in avg_data:
                    avg_data[k] += d[k]
        assert avg_data
        for p in avg_data.values():
            assert not p.isnan().any().cpu()
        return avg_data

    @classmethod
    def weighted_avg_for_scalar(
        cls,
        data_dict: MutableMapping[int, Message],
        weights: dict[int, float] | float,
        scalar_key: str,
    ) -> float:
        assert data_dict
        result: float = 0
        for worker_id, v in data_dict.items():
            weight = weights[worker_id] if isinstance(weights, dict) else weights
            assert 0 <= weight <= 1
            result += v.other_data[scalar_key] * weight
        return result

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: Message | None,
    ) -> bool:
        if worker_data is None:
            self.__skipped_workers.add(worker_id)
            return True
        self._all_worker_data[worker_id] = worker_data
        return True

    def aggregate_worker_data(self) -> Any:
        raise NotImplementedError()

    def clear_worker_data(self) -> None:
        self._all_worker_data.clear()
        self.__skipped_workers.clear()

    def exit(self) -> None:
        pass

import json
import os
from typing import Any

from cyy_naive_lib.log import log_info

from ..message import ParameterMessage
from .protocol import AggregationServerProtocol


class PerformanceMixin(AggregationServerProtocol):
    def __init__(self) -> None:
        super().__init__()
        self.__stat: dict = {}
        self.__keys: list = []
        self.__plateau: int = 0
        self.__acc_diff = 0.001
        self.__max_plateau: int = 5

    @property
    def performance_stat(self) -> dict:
        return self.__stat

    def _set_plateau_limit(self, max_plateau: int) -> None:
        self.__max_plateau = max_plateau

    def _set_accurary_difference(self, acc_diff: float) -> None:
        self.__acc_diff = acc_diff

    def _get_stat_key(self, message: ParameterMessage) -> Any:
        if message.is_initial:
            return 0
        return self.round_index

    def _set_stat(self, key: str, value: Any, message: ParameterMessage) -> None:
        stat_key = self._get_stat_key(message=message)
        if stat_key not in self.__keys:
            self.__keys.append(stat_key)
        self.__stat[stat_key][key] = value

    def record_performance_statistics(
        self,
        message: ParameterMessage,
    ) -> None:
        metric = self.get_metric(
            message.parameter, log_performance_metric=(not message.is_initial)
        )
        round_stat = {f"test_{k}": v for k, v in metric.items()}
        key = self._get_stat_key(message)
        assert key not in self.__stat
        self.__keys.append(key)
        self.__stat[key] = round_stat
        with open(
            os.path.join(self.save_dir, "round_record.json"),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(self.__stat, f)

    def get_test_accuracies(self) -> list[float]:
        return [self.performance_stat[k]["test_accuracy"] for k in self.__keys]

    def convergent(self) -> bool:
        if len(self.performance_stat) < 2:
            return False
        test_accuracies = self.get_test_accuracies()

        historical_max_acc = max(test_accuracies[0:-1])
        if test_accuracies[-1] > historical_max_acc + self.__acc_diff:
            self.__plateau = 0
            return False
        self.__plateau += 1
        log_info(
            "historical_max_acc is %s diff is %s, plateau is %s",
            historical_max_acc,
            historical_max_acc - test_accuracies[-1],
            self.__plateau,
        )
        return self.__plateau >= self.__max_plateau

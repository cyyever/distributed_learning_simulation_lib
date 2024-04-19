import json
import os
from typing import Any

from cyy_naive_lib.log import log_error

from ..message import ParameterMessage
from .protocol import AggregationServerProtocol


class PerformanceMixin(AggregationServerProtocol):
    def __init__(self) -> None:
        super().__init__()
        self.__stat: dict = {}
        self.__plateau = 0
        self.__max_acc = 0

    @property
    def performance_stat(self) -> dict:
        return self.__stat

    def _get_stat_key(self) -> Any:
        return self.round_index

    def _set_stat(self, key: str, value: Any) -> None:
        stat_key = self._get_stat_key()
        if stat_key == 1:
            if 0 in self.__stat and 1 not in self.__stat:
                stat_key = 0
        self.__stat[stat_key][key] = value

    def record_compute_stat(
        self,
        message: ParameterMessage,
    ) -> None:
        metric = self.get_metric(
            message.parameter, log_performance_metric=(not message.is_initial)
        )
        round_stat = {f"test_{k}": v for k, v in metric.items()}
        key = 0
        if not message.is_initial:
            key = self._get_stat_key()
        assert key not in self.__stat

        self.__stat[key] = round_stat
        with open(
            os.path.join(self.save_dir, "round_record.json"),
            "wt",
            encoding="utf8",
        ) as f:
            json.dump(self.__stat, f)

        max_acc = max(t["test_accuracy"] for t in self.__stat.values())
        if max_acc > self.__max_acc:
            self.__max_acc = max_acc

    def convergent(self) -> bool:
        max_acc = max(t["test_accuracy"] for t in self.performance_stat.values())
        diff = 0.001
        if max_acc > self.__max_acc + diff:
            self.__max_acc = max_acc
            self.__plateau = 0
            return False
        del max_acc
        log_error(
            "max acc is %s diff is %s",
            self.__max_acc,
            self.__max_acc
            - self.performance_stat[self._get_stat_key()]["test_accuracy"],
        )
        self.__plateau += 1
        log_error("plateau is %s", self.__plateau)
        if self.__plateau >= 5:
            return True
        return False

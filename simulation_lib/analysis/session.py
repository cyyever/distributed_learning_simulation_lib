import functools
import json
import os

import dill

from ..config import DistributedTrainingConfig


class Session:
    def __init__(self, session_dir: str | None = None) -> None:
        if session_dir is None:
            session_dir = os.getenv("SESSION_DIR")
        assert session_dir
        self.session_dir = session_dir

        with open(
            os.path.join(self.__server_dir, "round_record.json"), encoding="utf8"
        ) as f:
            self.round_record = json.load(f)
        self.round_record = {int(k): v for k, v in self.round_record.items()}
        with open(os.path.join(self.__server_dir, "config.pkl"), "rb") as f:
            self.config: DistributedTrainingConfig = dill.load(f)

        self.__worker_data: dict = {}

    @property
    def last_model_path(self) -> str:
        path = os.path.join(
            self.session_dir, "aggregated_model", f"round_{self.config.round}.pk"
        )
        assert os.path.exists(path)
        return path

    @property
    def __server_dir(self) -> str:
        server_dir = os.path.join(self.session_dir, "server")
        assert os.path.isdir(server_dir)
        return server_dir

    @property
    def worker_data(self) -> dict:
        if self.__worker_data:
            return self.__worker_data
        worker_data: dict = {}
        for root, dirs, __ in os.walk(self.session_dir):
            for name in dirs:
                if name.startswith("worker"):
                    worker_data[name] = {}
                    with open(
                        os.path.join(root, name, "hyper_parameter.pk"),
                        "rb",
                    ) as f:
                        worker_data[name]["hyper_parameter"] = dill.load(f)
                    if os.path.isfile(
                        os.path.join(root, name, "graph_worker_stat.json")
                    ):
                        with open(
                            os.path.join(root, name, "graph_worker_stat.json"),
                            encoding="utf8",
                        ) as f:
                            worker_data[name] = json.load(f)
        assert worker_data
        self.__worker_data = worker_data
        return self.__worker_data

    @functools.cached_property
    def rounds(self) -> list:
        return sorted(self.round_record.keys())

    @functools.cached_property
    def last_round(self) -> int:
        return self.rounds[-1]

    @functools.cached_property
    def last_test_acc(self) -> float:
        return self.round_record[self.last_round]["test_accuracy"]

    @functools.cached_property
    def mean_test_acc(self) -> float:
        total_acc = 0
        for r in self.round_record.values():
            total_acc += r["test_accuracy"]
        return total_acc / len(self.round_record)

import functools
import json
import os
from pathlib import Path
from typing import Any

import dill
from cyy_torch_toolbox import TensorDict

from .config import DistributedTrainingConfig


class Session:
    def __init__(self, session_dir: Path | str | None = None) -> None:
        if session_dir is None:
            env_dir = os.getenv("SESSION_DIR")
            if not env_dir:
                raise ValueError(
                    "session_dir not provided and SESSION_DIR environment variable is not set"
                )
            self.session_dir = Path(env_dir).resolve()
        else:
            self.session_dir = Path(session_dir).resolve()
        assert self.session_dir.is_dir()

        with open(self.server_dir / "round_record.json", encoding="utf8") as f:
            self.round_record = json.load(f)
        self.round_record = {int(k): v for k, v in self.round_record.items()}
        with open(self.server_dir / "config.pkl", "rb") as f:
            self.config: DistributedTrainingConfig = dill.load(f)

        self.__worker_data: dict[str, dict[str, Any]] = {}

    @property
    def last_model_path(self) -> Path:
        path = self.session_dir / "aggregated_model" / f"round_{self.config.round}.pk"
        assert path.exists()
        return path

    def get_last_model_parameters(self) -> TensorDict:
        with open(self.last_model_path, "rb") as f:
            return dill.load(f)

    @property
    def server_dir(self) -> Path:
        _server_dir = self.session_dir / "server"
        assert _server_dir.is_dir()
        return _server_dir

    def worker_dir(self, worker_index: int) -> Path:
        _worker_dir = self.session_dir / f"worker_{worker_index}"
        assert _worker_dir.is_dir()
        return _worker_dir

    @property
    def worker_data(self) -> dict[str, dict[str, Any]]:
        if self.__worker_data:
            return self.__worker_data
        worker_data: dict[str, dict[str, Any]] = {}
        for root, dirs, _ in self.session_dir.walk():
            for name in dirs:
                if name.startswith("worker"):
                    worker_data[name] = {}
                    stat_file = root / name / "graph_worker_stat.json"
                    if stat_file.is_file():
                        with open(stat_file, encoding="utf8") as f:
                            worker_data[name] = json.load(f)
                    with open(root / name / "hyper_parameter.pk", "rb") as f:
                        worker_data[name]["hyper_parameter"] = dill.load(f)
        assert worker_data
        self.__worker_data = worker_data
        return self.__worker_data

    @functools.cached_property
    def rounds(self) -> list[int]:
        return sorted(self.round_record.keys())

    @functools.cached_property
    def last_round(self) -> int:
        return self.rounds[-1]

    @functools.cached_property
    def last_test_acc(self) -> float:
        return self.round_record[self.last_round]["test_accuracy"]

    @functools.cached_property
    def mean_test_acc(self) -> float:
        return sum(r["test_accuracy"] for r in self.round_record.values()) / len(
            self.round_record
        )

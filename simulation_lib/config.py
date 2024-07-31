import datetime
import importlib
import os
import uuid
from typing import Any

import omegaconf
from cyy_naive_lib.log import log_debug, log_warning
from cyy_torch_toolbox import ClassificationDatasetCollection, Config
from cyy_torch_toolbox.device import get_device_memory_info

from .practitioner import Practitioner
from .sampler import get_dataset_collection_split


class DistributedTrainingConfig(Config):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.distributed_algorithm: str = ""
        self.algorithm_kwargs: dict = {}
        self.worker_number: int = 0
        self.round: int = 0
        self.dataset_sampling: str = "iid"
        self.dataset_sampling_kwargs: dict[str, Any] = {}
        self.endpoint_kwargs: dict = {}
        self.exp_name: str = ""
        self.log_file: str = ""
        self.enable_training_log: bool = False
        self.use_validation: bool = False
        self.worker_number_per_process: int = 0

    def load_config_and_process(self, conf: Any) -> None:
        self.load_config(conf)
        self.reset_session()
        import_dependencies(
            dataset_type=self.dc_config.dataset_kwargs.get("dataset_type", None)
        )

    def get_worker_number_per_process(self) -> int:
        if self.worker_number_per_process != 0:
            return self.worker_number_per_process

        memory_info = get_device_memory_info()
        refined_memory_info: dict = {}
        MB = 1024 * 1024
        GB = MB * 1024
        for device, info in memory_info.items():
            if info.total / GB >= 20:
                if info.free / GB < 5:
                    continue
            if info.used / info.total > 0.9:
                continue
            free_GB = int(info.free / GB)
            if free_GB == 0:
                continue
            refined_memory_info[device] = info.free
        assert refined_memory_info
        log_warning("Use devices %s", list(refined_memory_info.keys()))
        if self.worker_number <= len(refined_memory_info):
            return 1
        # small scale training
        if self.worker_number <= 50:
            return int(self.worker_number / len(refined_memory_info))
        total_bytes = sum(refined_memory_info.values())
        MB_per_worker = min(total_bytes / MB / self.worker_number, 10 * GB)
        log_debug(
            "MB_per_worker %s other %s",
            MB_per_worker,
            min(refined_memory_info.values()) / MB,
        )
        worker_number_per_process = int(
            min(refined_memory_info.values()) / MB / MB_per_worker
        )
        assert worker_number_per_process > 0
        return worker_number_per_process

    def reset_session(self) -> None:
        task_time = datetime.datetime.now()
        date_time = f"{task_time:%Y-%m-%d_%H_%M_%S}"
        dataset_name = self.dc_config.dataset_kwargs.get(
            "name", self.dc_config.dataset_name
        )
        dir_suffix = os.path.join(
            self.distributed_algorithm,
            (
                f"{dataset_name}_{self.dataset_sampling}"
                if isinstance(self.dataset_sampling, str)
                else f"{dataset_name}_{'_'.join(self.dataset_sampling)}"
            ),
            self.model_config.model_name,
            date_time,
            str(uuid.uuid4().int + os.getpid()),
        )
        if self.exp_name:
            dir_suffix = os.path.join(self.exp_name, dir_suffix)
        self.save_dir = os.path.join("session", dir_suffix)
        self.log_file = str(os.path.join("log", dir_suffix)) + ".log"

    def create_practitioners(self) -> set[Practitioner]:
        practitioners: set[Practitioner] = set()
        dataset_collection = self.create_dataset_collection()
        assert isinstance(dataset_collection, ClassificationDatasetCollection)
        sampler = get_dataset_collection_split(
            name=self.dataset_sampling,
            dataset_collection=dataset_collection,
            part_number=self.worker_number,
            **self.dataset_sampling_kwargs,
        )
        for practitioner_id in range(self.worker_number):
            practitioner = Practitioner(
                practitioner_id=practitioner_id,
            )
            practitioner.set_sampler(sampler=sampler)
            practitioners.add(practitioner)
        assert practitioners
        return practitioners


def load_config(conf_obj: Any, global_conf_path: str) -> DistributedTrainingConfig:
    config: DistributedTrainingConfig = DistributedTrainingConfig()
    while "dataset_name" not in conf_obj and len(conf_obj) == 1:
        conf_obj = next(iter(conf_obj.values()))
    result_conf = omegaconf.OmegaConf.load(global_conf_path)
    result_conf.merge_with(conf_obj)
    config.load_config_and_process(result_conf)
    return config


def load_config_from_file(
    config_file: str, global_conf_path: str
) -> DistributedTrainingConfig:
    return load_config(omegaconf.OmegaConf.load(config_file), global_conf_path)


import_result = {}


def import_dependencies(dataset_type: str | None = None) -> dict:
    global import_result
    if import_result:
        return import_result
    libs = ["cyy_torch_graph", "cyy_torch_text", "cyy_torch_vision"]
    if dataset_type is not None:
        match dataset_type.lower():
            case "graph":
                libs = ["cyy_torch_graph"]
            case "vision":
                libs = ["cyy_torch_vision"]
            case "text":
                libs = ["cyy_torch_text"]
            case _:
                raise NotImplementedError(dataset_type)
    for dependency in libs:
        try:
            importlib.import_module(dependency)
            import_result[dependency] = True
        except ModuleNotFoundError:
            pass
    return import_result

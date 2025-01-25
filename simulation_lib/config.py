import datetime
import importlib
import os
import uuid
from typing import Any

import omegaconf
from cyy_naive_lib.system_info import OSType, get_operating_system_type
from cyy_torch_toolbox import Config, MachineLearningPhase

from .context import get_worker_number_per_process as _get_worker_number_per_process
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

    def load_config_and_process(self, conf: Any, import_libs: bool = True) -> None:
        self.load_config(conf)
        self.reset_session()
        if not import_libs:
            return
        import_dependencies(
            dataset_type=self.dc_config.dataset_kwargs.get("dataset_type", None)
        )

    def get_worker_number_per_process(self) -> int:
        if self.worker_number_per_process == 0:
            self.worker_number_per_process = _get_worker_number_per_process(
                self.worker_number
            )
        return self.worker_number_per_process

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
        if get_operating_system_type() == OSType.Windows:
            dir_suffix = str(uuid.uuid4().int + os.getpid())
        self.save_dir = os.path.join("session", dir_suffix)
        self.log_file = str(os.path.join("log", dir_suffix)) + ".log"

    def create_practitioners(self) -> set[Practitioner]:
        practitioners: set[Practitioner] = set()
        dataset_collection = self.create_dataset_collection()
        assert dataset_collection.has_dataset(phase=MachineLearningPhase.Test)
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


def load_config(
    conf_obj: Any, global_conf_path: str, import_libs: bool = True
) -> DistributedTrainingConfig:
    config: DistributedTrainingConfig = DistributedTrainingConfig()
    while "dataset_name" not in conf_obj and len(conf_obj) == 1:
        conf_obj = next(iter(conf_obj.values()))
    result_conf = omegaconf.OmegaConf.load(global_conf_path)
    result_conf.merge_with(conf_obj)
    config.load_config_and_process(result_conf, import_libs=import_libs)
    return config


def load_config_from_file(
    config_file: str, global_conf_path: str
) -> DistributedTrainingConfig:
    return load_config(omegaconf.OmegaConf.load(config_file), global_conf_path)


import_result: dict = {}


def import_dependencies(dataset_type: str | None = None) -> None:
    if import_result:
        return
    libs = ["cyy_torch_text", "cyy_torch_vision"]
    if dataset_type is not None:
        match dataset_type.lower():
            case "vision":
                libs = ["cyy_torch_vision"]
            case "text":
                libs = ["cyy_torch_text"]
    for dependency in libs:
        try:
            importlib.import_module(dependency)
            import_result[dependency] = True
        except ModuleNotFoundError:
            pass

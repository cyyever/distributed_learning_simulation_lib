from pathlib import Path
from typing import cast

import torch
from cyy_naive_lib.storage import DataStorage
from cyy_preprocessing_pipeline import tensor_to
from cyy_torch_toolbox import ModelParameter


class ModelCache:
    __slots__ = ("__parameter",)

    def __init__(self) -> None:
        self.__parameter: DataStorage = DataStorage()

    @property
    def has_data(self) -> bool:
        return self.__parameter.has_data()

    @property
    def parameter(self) -> ModelParameter:
        return cast(ModelParameter, self.__parameter.data)

    def load_file(self, path: Path) -> None:
        self.__parameter = DataStorage(data_path=path)

    def cache_parameter(self, parameter: ModelParameter, path: Path) -> None:
        self.__parameter.set_data(
            {
                k: v.to(dtype=torch.float64)
                for k, v in tensor_to(parameter, device="cpu").items()
            }
        )
        self.__parameter.set_data_path(path)

    def get_parameter_diff(self, new_parameter: ModelParameter) -> ModelParameter:
        return {k: v - self.parameter[k] for k, v in new_parameter.items()}

    def add_parameter_diff(self, parameter_diff: ModelParameter, path: Path) -> None:
        self.__parameter.set_data_path(path)
        for k, v in self.parameter.items():
            self.parameter[k] = v + tensor_to(parameter_diff[k], device="cpu")
        self.__parameter.mark_new_data()

    def discard(self) -> None:
        self.__parameter.clear()

    def save(self) -> None:
        self.__parameter.save()

    def get_parameter_path(self) -> Path:
        self.__parameter.save()
        assert self.__parameter.data_path
        return Path(self.__parameter.data_path)

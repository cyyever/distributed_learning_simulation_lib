from cyy_naive_lib.storage import DataStorage
from cyy_torch_toolbox import ModelParameter, TensorDict, tensor_to


class ModelCache:
    def __init__(self) -> None:
        self.__parameter: DataStorage = DataStorage()

    @property
    def has_data(self) -> bool:
        return self.__parameter.has_data()

    @property
    def parameter(self) -> ModelParameter:
        return self.__parameter.data

    def load_file(self, path: str) -> None:
        self.__parameter = DataStorage(data_path=path)

    def cache_parameter_dict(self, parameter_dict: TensorDict, path: str) -> None:
        self.__parameter.set_data(tensor_to(parameter_dict, device="cpu"))
        self.__parameter.set_data_path(path)

    def get_parameter_diff(self, new_parameter_dict: TensorDict) -> ModelParameter:
        return {
            k: tensor_to(v, device="cpu") - self.parameter[k]
            for k, v in new_parameter_dict.items()
        }

    def add_parameter_diff(self, parameter_diff: ModelParameter, path: str) -> None:
        self.__parameter.set_data_path(path)
        for k, v in self.parameter.items():
            self.parameter[k] = v + tensor_to(parameter_diff[k], device="cpu")
        self.__parameter.mark_new_data()

    def discard(self) -> None:
        self.__parameter.clear()

    def save(self) -> None:
        self.__parameter.save()

    def get_parameter_path(self) -> str:
        self.__parameter.save()
        assert self.__parameter.data_path
        return self.__parameter.data_path

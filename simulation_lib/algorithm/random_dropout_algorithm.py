import random

from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import ModelParameter, cat_tensors_to_vector


class RandomDropoutAlgorithm:
    def __init__(self, dropout_rate) -> None:
        self.__dropout_rate = dropout_rate
        log_info("use dropout rate %s", self.__dropout_rate)

    def drop_parameter(self, parameter_dict: ModelParameter) -> ModelParameter:
        parameter_num = cat_tensors_to_vector(parameter_dict.values()).nelement()
        threshold = (1 - self.__dropout_rate) * parameter_num
        partial_parameter_num = 0
        parameter_names = list(parameter_dict.keys())
        random.shuffle(parameter_names)
        new_parameter_dict: dict = {}
        for k in parameter_names:
            if partial_parameter_num > threshold:
                break
            parameter = parameter_dict[k]
            if partial_parameter_num + parameter.nelement() > threshold:
                continue
            new_parameter_dict[k] = parameter
        log_info(
            "partial_parameter_num %s threshold %s",
            partial_parameter_num,
            threshold,
        )
        return new_parameter_dict

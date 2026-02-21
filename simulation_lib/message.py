import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from cyy_preprocessing_pipeline.tensor import recursive_tensor_op
from cyy_torch_toolbox import ModelParameter


@dataclass(kw_only=True, slots=True)
class Message:
    other_data: dict[str, Any] = field(default_factory=dict)
    in_round: bool = False
    end_training: bool = False
    aggregation_weight: float | None = None


@dataclass(kw_only=True, slots=True)
class ParameterMessageBase(Message):
    is_initial: bool = False


@dataclass(kw_only=True, slots=True)
class ParameterMessage(ParameterMessageBase):
    parameter: ModelParameter

    def complete(self, other_parameter: ModelParameter) -> None:
        for k, v in other_parameter.items():
            if k not in self.parameter:
                self.parameter[k] = v


@dataclass(kw_only=True, slots=True)
class DeltaParameterMessage(ParameterMessageBase):
    delta_parameter: ModelParameter
    old_parameter: ModelParameter | None = None
    new_parameter: ModelParameter | None = None

    def restore(self, parameter: ModelParameter) -> ParameterMessage:
        restored_parameter = copy.deepcopy(parameter)
        if self.old_parameter is not None:
            assert len(self.old_parameter) == len(restored_parameter)
            for k, v in self.old_parameter.items():
                assert (v.cpu() == restored_parameter[k]).all().item()
        assert len(self.delta_parameter) == len(parameter)

        for k, v in self.delta_parameter.items():
            restored_parameter[k] = restored_parameter[k].to(dtype=torch.float64) + v
            if self.new_parameter is not None:
                v2 = self.new_parameter[k].to(dtype=torch.float64, device="cpu")
                assert torch.allclose(v2, restored_parameter[k]), (
                    f"Restoration mismatch for key {k}: "
                    f"delta={v}, result={restored_parameter[k]}, expected={v2}"
                )

        msg = ParameterMessage(parameter=restored_parameter)
        for f in fields(msg):
            if f.name != "parameter" and hasattr(self, f.name):
                setattr(msg, f.name, getattr(self, f.name))
        return msg


@dataclass(kw_only=True, slots=True)
class FeatureMessage(Message):
    feature: torch.Tensor | None


@dataclass(kw_only=True, slots=True)
class MultipleWorkerMessage(Message):
    worker_data: Mapping[int, Message]


def get_message_size(msg: Message) -> int:
    cnt: int = 0

    def count(data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        nonlocal cnt
        cnt += data.element_size() * data.numel()
        return data

    recursive_tensor_op(msg, fun=count)
    assert cnt > 0
    return cnt

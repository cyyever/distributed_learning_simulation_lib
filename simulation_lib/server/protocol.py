from functools import cached_property
from typing import Protocol

from cyy_torch_toolbox.typing import TensorDict

from ..config import DistributedTrainingConfig
from ..message import ParameterMessage


class AggregationServerProtocol(Protocol):
    @property
    def config(self) -> DistributedTrainingConfig:
        ...

    async def get_metric(
        self,
        parameter_dict: TensorDict | ParameterMessage,
        log_performance_metric: bool = True,
    ) -> dict:
        ...

    @property
    def round_index(self) -> int:
        ...

    @cached_property
    def save_dir(self) -> str:
        ...

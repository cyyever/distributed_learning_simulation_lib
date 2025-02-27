from functools import cached_property
from typing import Protocol

from cyy_torch_toolbox import ModelParameter

from ..config import DistributedTrainingConfig
from ..message import ParameterMessage
from ..protocol import ExecutorProtocol


class AggregationServerProtocol(ExecutorProtocol, Protocol):
    @property
    def config(self) -> DistributedTrainingConfig: ...

    @property
    def worker_number(self) -> int: ...

    def get_metric(
        self,
        parameter: ModelParameter | ParameterMessage,
        log_performance_metric: bool = True,
    ) -> dict: ...

    @property
    def round_index(self) -> int: ...

    @cached_property
    def save_dir(self) -> str: ...

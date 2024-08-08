import copy
import os
import pickle
import time
from typing import Any

from cyy_naive_lib.log import log_debug, log_info
from cyy_naive_lib.topology import ServerEndpoint
from cyy_torch_toolbox import Inferencer, MachineLearningPhase, ModelParameter

from ..executor import Executor, ExecutorContext
from ..message import Message, ParameterMessage
from .round_selection_mixin import RoundSelectionMixin


class Server(Executor, RoundSelectionMixin):
    def __init__(self, task_id: int, endpoint: ServerEndpoint, **kwargs: Any) -> None:
        name: str = "server"
        if task_id is not None:
            name = f"server of {task_id}"
        super().__init__(**kwargs, name=name)
        RoundSelectionMixin.__init__(self)
        self._endpoint: ServerEndpoint = endpoint
        self.__tester: Inferencer | None = None

    @property
    def worker_number(self) -> int:
        return self.config.worker_number

    def get_tester(self) -> Inferencer:
        if self.__tester is not None:
            return self.__tester
        tester = self.config.create_inferencer(
            phase=MachineLearningPhase.Test, inherent_device=False
        )
        tester.dataset_collection.remove_dataset(phase=MachineLearningPhase.Training)
        tester.dataset_collection.remove_dataset(phase=MachineLearningPhase.Validation)
        tester.hook_config.summarize_executor = False
        self.__tester = tester
        return tester

    def load_parameter(self, tester: Inferencer, parameter: ModelParameter) -> None:
        tester.model_util.load_parameters(parameter)

    def get_metric(
        self,
        parameter: ModelParameter | ParameterMessage,
        log_performance_metric: bool = True,
    ) -> dict:
        if isinstance(parameter, ParameterMessage):
            parameter = parameter.parameter
        tester = self.get_tester()
        self.load_parameter(tester=tester, parameter=parameter)
        tester.model_util.disable_running_stats()
        tester.hook_config.log_performance_metric = log_performance_metric
        tester.hook_config.save_performance_metric = log_performance_metric
        batch_size: int | None = None
        if "server_batch_size" in tester.dataloader_kwargs:
            batch_size = tester.dataloader_kwargs["server_batch_size"]
            tester.remove_dataloader_kwargs("server_batch_size")
        elif "batch_number" in tester.dataloader_kwargs:
            batch_size = min(
                int(tester.dataset_size / tester.dataloader_kwargs["batch_number"]),
                100,
            )
        if batch_size is not None:
            assert batch_size > 0
            log_info("server uses batch_size %s", batch_size)
            tester.remove_dataloader_kwargs("batch_number")
            tester.update_dataloader_kwargs(batch_size=batch_size)
        if tester.has_hook_obj("performance_metric"):
            tester.performance_metric.clear_metric()
            metric: dict = tester.performance_metric.get_epoch_metrics(1)
            assert not metric
        tester.inference()
        metric = tester.performance_metric.get_epoch_metrics(1)
        assert metric
        tester.offload_from_device()
        return metric

    def start(self) -> None:
        ExecutorContext.set_name(self.name)
        with open(os.path.join(self.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        self._before_start()

        worker_set: set = set()
        while not self._stopped():
            if not worker_set:
                worker_set = set(range(self._endpoint.worker_num))
            assert self._endpoint.worker_num == self.config.worker_number
            for worker_id in copy.copy(worker_set):
                has_data: bool = self._endpoint.has_data(worker_id)
                if has_data:
                    log_debug(
                        "get result from %s worker_num %s",
                        worker_id,
                        self._endpoint.worker_num,
                    )
                    self._process_worker_data(
                        worker_id, self._endpoint.get(worker_id=worker_id)
                    )
                    worker_set.remove(worker_id)
            if worker_set:
                log_debug("wait workers %s", worker_set)

            if worker_set and not self._stopped():
                time.sleep(1)

        self._endpoint.close()
        self._server_exit()
        log_info("end server")

    def _before_start(self) -> None:
        pass

    def _server_exit(self) -> None:
        pass

    def _process_worker_data(self, worker_id: int, data: Message) -> None:
        raise NotImplementedError()

    def _stopped(self) -> bool:
        raise NotImplementedError()

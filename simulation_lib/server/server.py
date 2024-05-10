import asyncio
import copy
import os
import pickle
import random
from typing import Any

import torch
from cyy_naive_lib.log import log_debug
from cyy_naive_lib.topology.cs_endpoint import ServerEndpoint
from cyy_torch_toolbox import Inferencer, MachineLearningPhase
from cyy_torch_toolbox.typing import TensorDict

from ..executor import Executor
from ..message import Message, MultipleWorkerMessage, ParameterMessage


class Server(Executor):
    def __init__(self, task_id: int, endpoint: ServerEndpoint, **kwargs: Any) -> None:
        name: str = "server"
        if task_id is not None:
            name = f"server of {task_id}"
        super().__init__(**kwargs, name=name)
        self._endpoint: ServerEndpoint = endpoint

    @property
    def worker_number(self) -> int:
        return self.config.worker_number

    def get_tester(self) -> Inferencer:
        tester = self.config.create_inferencer(
            phase=MachineLearningPhase.Test, inherent_device=False
        )
        tester.dataset_collection.remove_dataset(phase=MachineLearningPhase.Training)
        tester.dataset_collection.remove_dataset(phase=MachineLearningPhase.Validation)
        tester.hook_config.summarize_executor = False
        return tester

    def load_parameter(self, tester: Inferencer, parameter_dict: TensorDict):
        tester.model_util.load_parameter_dict(parameter_dict)

    async def get_metric(
        self,
        parameter_dict: TensorDict | ParameterMessage,
        log_performance_metric: bool = True,
    ) -> dict:
        if isinstance(parameter_dict, ParameterMessage):
            parameter_dict = parameter_dict.parameter
        tester = self.get_tester()
        self.load_parameter(tester=tester, parameter_dict=parameter_dict)
        tester.model_util.disable_running_stats()
        tester.set_device_fun(self._get_device)
        tester.hook_config.log_performance_metric = log_performance_metric
        if "batch_number" in tester.dataloader_kwargs:
            batch_size = min(
                int(tester.dataset_size / tester.dataloader_kwargs["batch_number"]),
                100,
            )
            log_debug("batch_size %s", batch_size)
            tester.remove_dataloader_kwargs("batch_number")
            tester.update_dataloader_kwargs(batch_size=batch_size)
        if "num_neighbor" in tester.dataloader_kwargs:
            tester.update_dataloader_kwargs(num_neighbor=10)
        await tester.async_inference()
        metric: dict = tester.performance_metric.get_epoch_metrics(1)
        self._release_device_lock()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metric

    async def start(self) -> None:
        with open(os.path.join(self.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        await self._before_start()

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
                    await self._process_worker_data(
                        worker_id, self._endpoint.get(worker_id=worker_id)
                    )
                    worker_set.remove(worker_id)
            if worker_set:
                log_debug("wait workers %s", worker_set)

            if worker_set and not self._stopped():
                await asyncio.sleep(1)

        self._endpoint.close()
        self._server_exit()
        log_debug("end server")

    async def _before_start(self) -> None:
        pass

    def _server_exit(self) -> None:
        pass

    async def _process_worker_data(self, worker_id: int, data: Message) -> None:
        raise NotImplementedError()

    async def _before_send_result(self, result: Message) -> None:
        pass

    def _after_send_result(self, result: Message) -> None:
        pass

    async def _send_result(self, result: Message) -> None:
        await self._before_send_result(result=result)
        if isinstance(result, MultipleWorkerMessage):
            for worker_id, data in result.worker_data.items():
                self._endpoint.send(worker_id=worker_id, data=data)
        else:
            selected_workers = self._select_workers()
            log_debug("choose workers %s", selected_workers)
            if selected_workers:
                self._endpoint.broadcast(data=result, worker_ids=selected_workers)
            unselected_workers = set(range(self.worker_number)) - selected_workers
            if unselected_workers:
                self._endpoint.broadcast(data=None, worker_ids=unselected_workers)
        self._after_send_result(result=result)

    def _select_workers(self) -> set:
        if "random_client_number" in self.config.algorithm_kwargs:
            return set(
                random.sample(
                    list(range(self.worker_number)),
                    k=self.config.algorithm_kwargs["random_client_number"],
                )
            )
        return set(range(self.worker_number))

    def _stopped(self) -> bool:
        raise NotImplementedError()

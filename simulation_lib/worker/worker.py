import functools
from typing import Any, override

import dill
from cyy_naive_lib.log import log_debug
from cyy_naive_lib.topology import ClientEndpoint
from cyy_torch_toolbox import ExecutorHookPoint, Trainer

from ..context import ClientEndpointInCoroutine
from ..executor import Executor
from ..practitioner import Practitioner
from ..task_type import TaskIDType


class Worker(Executor):
    def __init__(
        self,
        task_id: TaskIDType,
        endpoint: ClientEndpoint,
        practitioner: Practitioner,
        single_task: bool = False,
        **kwargs: Any,
    ) -> None:
        worker_id = practitioner.worker_id
        name = (
            f"worker {worker_id}" if single_task else f"worker {worker_id} of {task_id}"
        )
        super().__init__(name=name, task_id=task_id, **kwargs)
        self._practitioner: Practitioner = practitioner
        self._endpoint: ClientEndpoint = endpoint
        if isinstance(self._endpoint, ClientEndpointInCoroutine):
            self._endpoint.replace_context(context=self.context)
        self._round_index = 0
        self._force_stop = False
        self._in_after_training: bool = False
        self.__trainer: Trainer | None = None

    @property
    def endpoint(self) -> ClientEndpoint:
        return self._endpoint

    @property
    def round_index(self) -> int:
        return self._round_index

    @property
    def worker_id(self) -> int:
        return self._practitioner.worker_id

    @property
    def trainer(self) -> Trainer:
        if self.__trainer is None:
            self.__trainer = self.__new_trainer()
        return self.__trainer

    def clear_trainer(self) -> None:
        self.__trainer = None

    def __new_trainer(self) -> Trainer:
        if "server_batch_size" in self.config.trainer_config.dataloader_kwargs:
            self.config.trainer_config.dataloader_kwargs.pop("server_batch_size")
        return self._practitioner.create_trainer(self.config)

    def _before_training(self) -> None:
        self.trainer.set_device_fun(
            functools.partial(
                self.context.get_device,
                lock_callback=lambda: self.trainer.append_named_hook(
                    ExecutorHookPoint.AFTER_BATCH,
                    "_release_device_lock",
                    self._release_device_lock,
                ),
            )
        )
        self.trainer.hook_config.log_performance_metric = (
            self.config.enable_training_log
        )
        self.trainer.hook_config.save_performance_metric = False
        self.trainer.disable_hook("batch_loss_logger")

    def _after_training(self) -> None:
        self._in_after_training = True
        with open(self.save_dir / "hyper_parameter.pk", "wb") as f:
            dill.dump(
                self.trainer.hyper_parameter,
                f,
            )

    def _stopped(self) -> bool:
        return self._round_index > self.config.round or self._force_stop

    def pause(self, in_round: bool = False) -> None:
        if not in_round:
            self.trainer.offload_from_device()
        self.context.release_device_lock()

    def _release_device_lock(self, *args, **kwargs) -> None:
        self.context.release_device_lock()
        self.trainer.remove_named_hook("_release_device_lock")

    def _train(self, first_training: bool) -> None:
        if not first_training:
            self.trainer.hook_config.summarize_executor = False
        self.trainer.set_visualizer_prefix(prefix=f"round: {self._round_index},")
        self.trainer.train()

    @override
    def start(self) -> None:
        first_training: bool = True
        self._round_index = 1
        self._force_stop = False
        while not self._stopped():
            with self.context:
                if first_training:
                    self._before_training()
                    # in case the worker changes round number
                    if self._stopped():
                        break
                self._train(first_training=first_training)
                first_training = False
                self._round_index += 1
        with self.context:
            log_debug("end training")
            self._after_training()
            self.endpoint.send(None)
            log_debug("end worker")

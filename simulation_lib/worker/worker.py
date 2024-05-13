import functools
import os
from functools import cached_property
from typing import Any

import dill
from cyy_naive_lib.log import log_debug
from cyy_naive_lib.topology.endpoint import Endpoint
from cyy_torch_toolbox import ExecutorHookPoint, Trainer

from ..executor import Executor, ExecutorContext
from ..practitioner import Practitioner


class Worker(Executor):
    def __init__(
        self,
        task_id: int | None,
        endpoint: Endpoint,
        practitioner: Practitioner,
        **kwargs: Any,
    ) -> None:
        worker_id = practitioner.worker_id
        name = f"worker {worker_id}"
        if task_id is not None:
            name = f"worker {worker_id} of {task_id}"
        super().__init__(name=name, **kwargs)
        self.__practitioner: Practitioner = practitioner
        self._endpoint = endpoint
        self._round_index = 0
        self._force_stop = False

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def round_index(self):
        return self._round_index

    @property
    def worker_id(self):
        return self.__practitioner.worker_id

    @cached_property
    def trainer(self) -> Trainer:
        return self.__new_trainer()

    def __new_trainer(self) -> Trainer:
        return self.__practitioner.create_trainer(self.config)

    def _offload_from_device(self) -> None:
        self.trainer.offload_from_device()

    async def _before_training(self) -> None:
        pass

    def _after_training(self) -> None:
        with open(os.path.join(self.save_dir, "hyper_parameter.pk"), "wb") as f:
            dill.dump(
                self.trainer.hyper_parameter,
                f,
            )

    def _stopped(self) -> bool:
        return self._round_index > self.config.round or self._force_stop

    def pause(self) -> None:
        self.trainer.wait_stream()
        self._release_device_lock()

    async def start(self, **kwargs: Any) -> None:
        first_training: bool = True
        self._round_index = 1
        self._force_stop = False
        while not self._stopped():
            # in case worker changes round number
            async with ExecutorContext(self.name):
                if first_training:
                    await self._before_training()
                    first_training = False
                    # in case worker changes round number
                    if self._stopped():
                        break
                    self.trainer.set_device_fun(
                        functools.partial(
                            self._get_device,
                            lock_callback=lambda: self.trainer.append_named_hook(
                                ExecutorHookPoint.AFTER_BATCH,
                                "_release_device_lock",
                                self._release_device_lock,
                            ),
                        )
                    )
                else:
                    self.trainer.hook_config.summarize_executor = False
                self.trainer.hook_config.log_performance_metric = (
                    self.config.enable_training_log
                )
                self.trainer.disable_hook("batch_loss_logger")
                self.trainer.set_visualizer_prefix(
                    prefix=f"round: {self._round_index},"
                )
                await self.trainer.async_train(
                    **kwargs,
                )
                self._round_index += 1
        async with ExecutorContext(self.name):
            log_debug("finish worker")
            self.endpoint.close()
            log_debug("close endpoint")
            self._after_training()
            log_debug("end worker")

import functools
import os

import dill
from cyy_torch_toolbox import ExecutorHookPoint, Trainer

from .worker_base import WorkerBase


class Worker(WorkerBase):
    __trainer: Trainer | None = None

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

    def pause(self, in_round: bool = False) -> None:
        if not in_round:
            self.trainer.offload_from_device()
        super().pause()

    def _after_training(self) -> None:
        with open(os.path.join(self.save_dir, "hyper_parameter.pk"), "wb") as f:
            dill.dump(
                self.trainer.hyper_parameter,
                f,
            )

    def _release_device_lock(self, *args, **kwargs) -> None:
        self.context.release_device_lock()
        self.trainer.remove_named_hook("_release_device_lock")

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

    def _train(self, first_training: bool) -> None:
        if not first_training:
            self.trainer.hook_config.summarize_executor = False
        self.trainer.hook_config.log_performance_metric = (
            self.config.enable_training_log
        )
        self.trainer.hook_config.save_performance_metric = False
        self.trainer.disable_hook("batch_loss_logger")
        self.trainer.set_visualizer_prefix(prefix=f"round: {self._round_index},")
        self.trainer.train()

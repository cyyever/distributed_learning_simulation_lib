from typing import Any

from cyy_naive_lib.log import log_debug
from cyy_naive_lib.topology import Endpoint

from ..executor import Executor, ExecutorContext
from ..practitioner import Practitioner


class WorkerBase(Executor):
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

    def _before_training(self) -> None:
        pass

    def _after_training(self) -> None:
        pass

    def _train(self, first_training: bool, training_kwargs: dict) -> None:
        raise NotImplementedError()

    def _stopped(self) -> bool:
        return self._round_index > self.config.round or self._force_stop

    def pause(self) -> None:
        self._release_device_lock()

    def start(self, **kwargs: Any) -> None:
        first_training: bool = True
        self._round_index = 1
        self._force_stop = False
        while not self._stopped():
            with ExecutorContext(self.name):
                if first_training:
                    self._before_training()
                    first_training = False
                    # in case the worker changes round number
                    if self._stopped():
                        break
                self._train(first_training=first_training, training_kwargs=kwargs)
                self._round_index += 1
        with ExecutorContext(self.name):
            log_debug("finish worker")
            self.endpoint.close()
            log_debug("close endpoint")
            self._after_training()
            log_debug("end worker")

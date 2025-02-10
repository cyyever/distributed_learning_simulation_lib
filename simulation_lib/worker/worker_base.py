from typing import Any

from cyy_naive_lib.log import log_debug
from cyy_naive_lib.topology import Endpoint

from ..executor import Executor
from ..practitioner import Practitioner
from ..task import OptionalTaskIDType


class WorkerBase(Executor):
    def __init__(
        self,
        task_id: OptionalTaskIDType,
        endpoint: Endpoint,
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
        self._endpoint: Endpoint = endpoint
        self._round_index = 0
        self._force_stop = False

    @property
    def endpoint(self) -> Endpoint:
        return self._endpoint

    @property
    def round_index(self):
        return self._round_index

    @property
    def worker_id(self):
        return self._practitioner.worker_id

    def _before_training(self) -> None:
        pass

    def _after_training(self) -> None:
        pass

    def _train(self, first_training: bool, training_kwargs: dict) -> None:
        raise NotImplementedError()

    def _stopped(self) -> bool:
        return self._round_index > self.config.round or self._force_stop

    def pause(self, in_round: bool = False) -> None:
        self.context.release_device_lock()

    def start(self, **kwargs: Any) -> None:
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
                self._train(first_training=first_training, training_kwargs=kwargs)
                first_training = False
                self._round_index += 1
        with self.context:
            log_debug("end training")
            self._after_training()
            self.endpoint.send(None)
            log_debug("end worker")

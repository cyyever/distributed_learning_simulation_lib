import os
import pickle
from typing import Any

from cyy_naive_lib.log import log_debug, log_info
from cyy_torch_toolbox import Inferencer, ModelParameter
from cyy_torch_toolbox.tensor import tensor_to

from ..algorithm.aggregation_algorithm import AggregationAlgorithm
from ..message import (DeltaParameterMessage, Message, MultipleWorkerMessage,
                       ParameterMessage, ParameterMessageBase)
from ..util.model_cache import ModelCache
from .performance_mixin import PerformanceMixin
from .round_selection_mixin import RoundSelectionMixin
from .server import Server


class AggregationServer(Server, PerformanceMixin, RoundSelectionMixin):
    def __init__(self, algorithm: AggregationAlgorithm, **kwargs: Any) -> None:
        Server.__init__(self, **kwargs)
        PerformanceMixin.__init__(self)
        RoundSelectionMixin.__init__(self)
        self._round_index: int = 1
        self._compute_stat: bool = True
        self._stop = False
        self.__model_cache: ModelCache = ModelCache()
        self.__worker_flag: set = set()
        algorithm.set_config(self.config)
        self.__algorithm: AggregationAlgorithm = algorithm
        self._need_init_performance = False

    @property
    def early_stop(self) -> bool:
        return self.config.algorithm_kwargs.get("early_stop", False)

    @property
    def algorithm(self):
        return self.__algorithm

    @property
    def round_index(self) -> int:
        return self._round_index

    def get_tester(self) -> Inferencer:
        tester = super().get_tester()
        tester.set_visualizer_prefix(f"round: {self.round_index},")
        return tester

    def __get_init_model(self) -> ModelParameter:
        parameter: ModelParameter = {}
        init_global_model_path = self.config.algorithm_kwargs.get(
            "global_model_path", None
        )
        if init_global_model_path is not None:
            with open(os.path.join(init_global_model_path), "rb") as f:
                parameter = pickle.load(f)
        else:
            parameter = self.get_tester().model_util.get_parameters()
        return parameter

    @property
    def distribute_init_parameters(self) -> bool:
        return self.config.algorithm_kwargs.get("distribute_init_parameters", True)

    def _before_start(self) -> None:
        if self.distribute_init_parameters:
            self._send_result(
                ParameterMessage(
                    in_round=True, parameter=self.__get_init_model(), is_initial=True
                )
            )

    def _send_result(self, result: Message) -> None:
        self._before_send_result(result=result)
        if isinstance(result, MultipleWorkerMessage):
            for worker_id, data in result.worker_data.items():
                self._endpoint.send(worker_id=worker_id, data=data)
        else:
            selected_workers = self.select_workers()
            if len(selected_workers) < self.config.worker_number:
                log_info("choose workers %s", selected_workers)
            if selected_workers:
                self._endpoint.broadcast(data=result, worker_ids=selected_workers)
            unselected_workers = set(range(self.worker_number)) - selected_workers
            if unselected_workers:
                self._endpoint.broadcast(data=None, worker_ids=unselected_workers)
        self._after_send_result(result=result)

    def _server_exit(self) -> None:
        self.__algorithm.exit()

    def _process_worker_data(self, worker_id: int, data: Message | None) -> None:
        assert 0 <= worker_id < self.worker_number
        log_debug("get data %s from worker %s", type(data), worker_id)
        if data is not None:
            if data.end_training:
                self._stop = True
                if not isinstance(data, ParameterMessageBase):
                    return

            old_parameter = self.__model_cache.parameter
            match data:
                case DeltaParameterMessage():
                    assert old_parameter is not None
                    data.delta_parameter = tensor_to(data.delta_parameter, device="cpu")
                    data = data.restore(old_parameter)
                case ParameterMessage():
                    if old_parameter is not None:
                        data.complete(old_parameter)
                    data.parameter = tensor_to(data.parameter, device="cpu")
        self.__algorithm.process_worker_data(worker_id=worker_id, worker_data=data)
        self.__worker_flag.add(worker_id)
        if len(self.__worker_flag) == self.worker_number:
            result = self._aggregate_worker_data()
            self._send_result(result)
            self.__worker_flag.clear()
        else:
            log_debug(
                "we have %s committed, and we need %s workers,skip",
                len(self.__worker_flag),
                self.worker_number,
            )

    def _aggregate_worker_data(self) -> Any:
        self.__algorithm.set_old_parameter(self.__model_cache.parameter)
        return self.__algorithm.aggregate_worker_data()

    def _before_send_result(self, result: Message) -> None:
        if not isinstance(result, ParameterMessageBase):
            return
        assert isinstance(result, ParameterMessage)
        if self._need_init_performance:
            assert self.distribute_init_parameters
        if self._need_init_performance and result.is_initial:
            self.record_performance_statistics(result)
        elif self._compute_stat and not result.is_initial and not result.in_round:
            self.record_performance_statistics(result)
            if not result.end_training and self.early_stop and self.convergent():
                log_info("stop early")
                self._stop = True
                result.end_training = True
        elif result.end_training:
            self.record_performance_statistics(result)
        model_path = os.path.join(
            self.config.save_dir,
            "aggregated_model",
            f"round_{self.round_index}.pk",
        )
        self.__model_cache.cache_parameter(result.parameter, model_path)

    def _after_send_result(self, result: Any) -> None:
        if not result.in_round:
            self._round_index += 1
        self.__algorithm.clear_worker_data()
        if result.end_training or self._stopped():
            assert self.__model_cache.has_data
            self.__model_cache.save()

    def _stopped(self) -> bool:
        return self.round_index > self.config.round or self._stop

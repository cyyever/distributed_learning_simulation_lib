import os
from typing import Any

import torch
from cyy_naive_lib.log import log_debug, log_info
from cyy_torch_toolbox import (
    ExecutorHookPoint,
    MachineLearningPhase,
    ModelParameter,
    StopExecutingException,
    TensorDict,
    tensor_to,
)
from cyy_torch_toolbox.hook.keep_model import KeepModelHook

from ..message import (
    DeltaParameterMessage,
    Message,
    ParameterMessage,
    ParameterMessageBase,
)
from ..util import ModelCache, load_parameters
from .client import ClientMixin
from .worker import Worker


class AggregationWorker(Worker, ClientMixin):
    def __init__(self, **kwargs: Any) -> None:
        Worker.__init__(self, **kwargs)
        ClientMixin.__init__(self)
        self._aggregation_time: ExecutorHookPoint = ExecutorHookPoint.AFTER_EXECUTE
        self._reuse_learning_rate: bool = False
        self.__choose_model_by_validation: bool | None = None
        self._send_parameter_diff: bool = False
        self._keep_model_cache: bool = False
        self._send_loss: bool = False
        self._model_cache: ModelCache = ModelCache()
        self._model_loading_fun = None

    @property
    def model_cache(self) -> ModelCache:
        return self._model_cache

    @property
    def distribute_init_parameters(self) -> bool:
        return self.config.algorithm_kwargs.get("distribute_init_parameters", True)

    def _before_training(self) -> None:
        super()._before_training()
        self.trainer.dataset_collection.remove_dataset(phase=MachineLearningPhase.Test)
        choose_model_by_validation = self.__choose_model_by_validation
        if choose_model_by_validation is None:
            choose_model_by_validation = self.config.hyper_parameter_config.epoch > 1
        if choose_model_by_validation:
            self.enable_choosing_model_by_validation()
        else:
            self.disable_choosing_model_by_validation()
        if not self.__choose_model_by_validation and not self.config.use_validation:
            # Skip Validation to speed up training
            self.trainer.dataset_collection.remove_dataset(
                phase=MachineLearningPhase.Validation
            )
        # load initial parameters
        if self.distribute_init_parameters:
            self.__get_result_from_server()
            if self._stopped():
                return
        self._register_aggregation()

    def _register_aggregation(self) -> None:
        log_debug("use aggregation_time %s", self._aggregation_time)
        self.trainer.remove_named_hook(name="aggregation")

        def __aggregation_impl(**kwargs) -> None:
            if not self._stopped():
                self._aggregation(sent_data=self._get_sent_data(), **kwargs)

        self.trainer.append_named_hook(
            self._aggregation_time,
            "aggregation",
            __aggregation_impl,
        )

    def _aggregation(self, sent_data: Message, **kwargs: Any) -> None:
        self._send_data_to_server(sent_data)
        self.__get_result_from_server()

    def enable_choosing_model_by_validation(self) -> None:
        self.__choose_model_by_validation = True
        hook = KeepModelHook()
        hook.keep_best_model = True
        assert self.trainer.dataset_collection.has_dataset(
            phase=MachineLearningPhase.Validation
        )
        self.trainer.remove_hook("keep_model_hook")
        self.trainer.append_hook(hook, "keep_model_hook")

    def disable_choosing_model_by_validation(self) -> None:
        self.__choose_model_by_validation = False
        self.trainer.remove_hook("keep_model_hook")

    @property
    def best_model_hook(self) -> KeepModelHook | None:
        if not self.trainer.has_hook_obj("keep_model_hook"):
            return None
        hook = self.trainer.get_hook("keep_model_hook")
        assert isinstance(hook, KeepModelHook)
        return hook

    def _get_parameters(self) -> TensorDict:
        return self.trainer.model_util.get_parameters()

    def _get_sent_data(self) -> ParameterMessageBase:
        if self.__choose_model_by_validation:
            assert self.best_model_hook is not None
            parameter = self.best_model_hook.best_model["parameter"]
            best_epoch = self.best_model_hook.best_model["epoch"]
            log_debug("use best model best_epoch %s", best_epoch)
        else:
            parameter = self._get_parameters()
            best_epoch = self.trainer.hyper_parameter.epoch
            log_debug(
                "use best model best_epoch %s acc %s parameter size %s",
                best_epoch,
                self.trainer.performance_metric.get_epoch_metric(
                    best_epoch, "accuracy"
                ),
                len(parameter),
            )
        parameter = tensor_to(parameter, device="cpu", dtype=torch.float64)
        other_data = {}
        if self._send_loss:
            other_data["training_loss"] = (
                self.trainer.performance_metric.get_epoch_metric(best_epoch, "loss")
            )
            assert other_data["training_loss"] is not None

        message: ParameterMessageBase = ParameterMessage(
            aggregation_weight=self.trainer.dataset_size,
            parameter=parameter,
            other_data=other_data,
        )
        if self._send_parameter_diff:
            assert self._model_cache.has_data
            message = DeltaParameterMessage(
                aggregation_weight=self.trainer.dataset_size,
                other_data=other_data,
                # old_parameter=self._model_cache.parameter,
                # new_parameter=parameter,
                delta_parameter=self._model_cache.get_parameter_diff(parameter),
            )
        if not self._keep_model_cache:
            self._model_cache.discard()
        return message

    def _load_result_from_server(self, result: Message) -> None:
        model_path = os.path.join(
            self.save_dir, "aggregated_model", f"round_{self.round_index}.pk"
        )
        parameter: ModelParameter = {}
        match result:
            case ParameterMessage():
                parameter = result.parameter
                if self._keep_model_cache or self._send_parameter_diff:
                    self._model_cache.cache_parameter(result.parameter, path=model_path)
            case DeltaParameterMessage():
                assert self._model_cache.has_data
                self._model_cache.add_parameter_diff(
                    result.delta_parameter, path=model_path
                )
                parameter = self._model_cache.parameter
            case _:
                raise NotImplementedError()
        load_parameters(
            trainer=self.trainer,
            parameter=parameter,
            reuse_learning_rate=self._reuse_learning_rate,
            loading_fun=self._model_loading_fun,
        )
        if result.end_training:
            self._force_stop = True
            raise StopExecutingException()

    def pause(self, in_round: bool = False) -> None:
        if not in_round:
            if self._model_cache.has_data:
                if self._keep_model_cache:
                    self._model_cache.save()
                else:
                    self._model_cache.discard()
            if self.best_model_hook is not None:
                self.best_model_hook.clear()
            if not self._reuse_learning_rate:
                self.trainer.remove_model()
        super().pause(in_round=in_round)

    def __get_result_from_server(self) -> None:
        while True:
            result = super()._get_data_from_server()
            log_debug("get result from server %s", type(result))
            if result is None:
                log_info("skip round %s", self.round_index)
                self._send_data_to_server(None)
                self._round_index += 1
                if self._stopped():
                    return
                continue
            self._load_result_from_server(result=result)
            break
        return

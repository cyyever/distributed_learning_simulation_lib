import os
from typing import Any

from cyy_naive_lib.log import log_debug, log_info
from cyy_torch_toolbox import (ExecutorHookPoint, MachineLearningPhase,
                               StopExecutingException)
from cyy_torch_toolbox.hook.keep_model import KeepModelHook

from ..message import (DeltaParameterMessage, Message, ParameterMessage,
                       ParameterMessageBase)
from ..util import ModelCache, load_parameters
from .client import Client


class AggregationWorker(Client):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._aggregation_time: ExecutorHookPoint = ExecutorHookPoint.AFTER_EXECUTE
        self._reuse_learning_rate: bool = False
        self.__choose_model_by_validation: bool | None = None
        self._send_parameter_diff: bool = True
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

    async def _before_training(self) -> None:
        await super()._before_training()
        self.trainer.dataset_collection.remove_dataset(phase=MachineLearningPhase.Test)
        if self.__choose_model_by_validation is None:
            if (
                self.config.hyper_parameter_config.epoch > 1
                and self.config.dataset_sampling == "iid"
            ):
                self.enable_choosing_model_by_validation()
            else:
                self.disable_choosing_model_by_validation()
        if not self.__choose_model_by_validation and not self.config.use_validation:
            # Skip Validation to speed up training
            self.trainer.dataset_collection.remove_dataset(
                phase=MachineLearningPhase.Validation
            )
        self.trainer.offload_from_device()
        # load initial parameters
        if self.distribute_init_parameters:
            await self.__get_result_from_server()
            if self._stopped():
                return
        self._register_aggregation()

    def _register_aggregation(self) -> None:
        log_debug("use aggregation_time %s", self._aggregation_time)
        self.trainer.remove_named_hook(name="aggregation")

        async def __aggregation_impl(**kwargs) -> None:
            if not self._stopped():
                await self._aggregation(sent_data=self._get_sent_data(), **kwargs)

        self.trainer.append_named_hook(
            self._aggregation_time,
            "aggregation",
            __aggregation_impl,
        )

    async def _aggregation(self, sent_data: Message, **kwargs: Any) -> None:
        self.send_data_to_server(sent_data)
        self._offload_from_device()
        await self.__get_result_from_server()

    def enable_choosing_model_by_validation(self) -> None:
        self.__choose_model_by_validation = True
        hook = KeepModelHook()
        hook.keep_best_model = True
        assert self.trainer.dataset_collection.has_dataset(
            phase=MachineLearningPhase.Validation
        )
        self.trainer.append_hook(hook, "keep_model_hook")

    def disable_choosing_model_by_validation(self) -> None:
        self.__choose_model_by_validation = False
        if self.best_model_hook is not None:
            self.trainer.remove_hook("keep_model_hook")

    @property
    def best_model_hook(self) -> KeepModelHook | None:
        if not self.trainer.has_hook_obj("keep_model_hook"):
            return None
        hook = self.trainer.get_hook("keep_model_hook")
        assert isinstance(hook, KeepModelHook)
        return hook

    def _get_sent_data(self) -> ParameterMessageBase:
        if self.__choose_model_by_validation:
            log_debug("use best model")
            assert self.best_model_hook is not None
            parameter = self.best_model_hook.best_model["parameter"]
            best_epoch = self.best_model_hook.best_model["epoch"]
        else:
            parameter = self.trainer.model_util.get_parameter_dict()
            best_epoch = self.trainer.hyper_parameter.epoch
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
                delta_parameter=self._model_cache.get_parameter_diff(parameter),
                other_data=other_data,
            )
        if not self._keep_model_cache:
            self._model_cache.discard()
        return message

    def _load_result_from_server(self, result: Message) -> None:
        model_path = os.path.join(
            self.save_dir, "aggregated_model", f"round_{self.round_index}.pk"
        )
        parameter_dict = {}
        match result:
            case ParameterMessage():
                parameter_dict = result.parameter
                if self._keep_model_cache or self._send_parameter_diff:
                    self._model_cache.cache_parameter_dict(
                        result.parameter, path=model_path
                    )
            case DeltaParameterMessage():
                assert self._model_cache.has_data
                self._model_cache.add_parameter_diff(
                    result.delta_parameter, path=model_path
                )
                parameter_dict = self._model_cache.parameter_dict
            case _:
                raise NotImplementedError()
        load_parameters(
            trainer=self.trainer,
            parameter_dict=parameter_dict,
            reuse_learning_rate=self._reuse_learning_rate,
            loading_fun=self._model_loading_fun,
        )
        if result.end_training:
            self._force_stop = True
            raise StopExecutingException()

    def _offload_from_device(self, in_round: bool = False) -> None:
        if self._model_cache.has_data:
            if self._keep_model_cache:
                self._model_cache.save()
            else:
                self._model_cache.discard()
        if self.best_model_hook is not None:
            assert not in_round
            self.best_model_hook.clear()
        super()._offload_from_device()

    async def __get_result_from_server(self) -> None:
        while True:
            result = await super()._get_data_from_server()
            log_debug("get result from server %s", type(result))
            if result is None:
                log_info("skip round %s", self.round_index)
                self.send_data_to_server(None)
                self._round_index += 1
                if self._stopped():
                    return
                continue
            self._load_result_from_server(result=result)
            break
        return

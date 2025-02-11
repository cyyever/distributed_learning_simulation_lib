from collections.abc import Callable

from cyy_naive_lib.log import log_debug
from cyy_torch_toolbox import ModelParameter, Trainer

from .model_cache import ModelCache


def __reset_optimizer_parameters(trainer: Trainer) -> None:
    optimizer = trainer.get_optimizer()
    assert len(optimizer.param_groups) == 1
    old_param_group = optimizer.param_groups[0]
    optimizer.param_groups.clear()
    optimizer.state.clear()
    optimizer.add_param_group({"params": trainer.model.parameters()})
    for k, v in old_param_group.items():
        if k not in "params":
            optimizer.param_groups[0][k] = v
            log_debug("reuse parameter property %s", k)


def load_parameters(
    trainer: Trainer,
    parameter: ModelParameter,
    reuse_learning_rate: bool,
    loading_fun: Callable | None = None,
) -> None:
    if loading_fun is not None:
        loading_fun(parameter)
    else:
        trainer.model_util.load_parameters(parameter)
    if reuse_learning_rate:
        __reset_optimizer_parameters(trainer=trainer)
    else:
        trainer.remove_optimizer()
    trainer.model_util.disable_running_stats()


__all__ = ["ModelCache", "load_parameters"]

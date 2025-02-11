import copy
import os
from collections.abc import Callable

from cyy_naive_lib.log import add_file_handler, log_debug, log_info
from cyy_naive_lib.time_counter import TimeCounter

from .algorithm_repository import get_worker_config
from .config import DistributedTrainingConfig
from .context import (
    ConcurrentFederatedLearningContext,
    FederatedLearningContext,
)
from .task import OptionalTaskIDType, TaskIDType, get_task_id
from .worker import Worker

# we use these environment variables to save memory in large-scale training
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["USE_THREAD_DATALOADER"] = "1"


def start_server(
    context: FederatedLearningContext, server_config: dict, single_task: bool
) -> dict:
    server = server_config["constructor"](context=context, single_task=single_task)
    log_debug("context id %d", id(context))

    server.start()
    log_info("stop server")

    res: dict = {}
    if hasattr(server.algorithm, "shapley_values"):
        res["sv"] = server.algorithm.shapley_values
    res |= {"performance": server.performance_stat}
    return res


def run_worker(constructor: Callable, **kwargs) -> None:
    worker: Worker = constructor(**kwargs)
    worker.start()


def start_workers(
    context: FederatedLearningContext,
    worker_configs: list[dict],
) -> None:
    assert isinstance(context, FederatedLearningContext)
    assert worker_configs

    log_debug(
        "run %s workers in the same process",
        len(worker_configs),
    )
    context.submit_batch(batch_fun=run_worker, kwargs_list=worker_configs)


concurrent_context = ConcurrentFederatedLearningContext()
task_results: dict = {}


def train(
    config: DistributedTrainingConfig,
    practitioners: None | set = None,
    single_task: bool = False,
) -> OptionalTaskIDType:
    # we need to deepcopy config for concurrent training
    config = copy.deepcopy(config)
    practitioners = copy.deepcopy(practitioners)
    config.reset_session()
    config.apply_global_config()
    timer = TimeCounter()
    task_id = get_task_id()
    if practitioners is None:
        add_file_handler(config.log_file)
    worker_config = get_worker_config(
        config, task_id=task_id, practitioners=practitioners
    )
    context = worker_config.pop("context")
    assert isinstance(context, FederatedLearningContext)
    server_config = worker_config.get("server", None)
    assert server_config is not None
    context.submit(start_server, server_config=server_config, single_task=single_task)
    for worker_configs in worker_config["worker"]:
        for cfg in worker_configs:
            cfg["single_task"] = single_task
        start_workers(context=context, worker_configs=worker_configs)
    concurrent_context.add_context(
        task_id=task_id,
        context=context,
        config=config,
        practitioner_ids={practitioner.id for practitioner in practitioners}
        if practitioners is not None
        else None,
    )
    if practitioners is not None:
        return task_id
    concurrent_context.wait_results(timeout=None)
    concurrent_context.release()
    log_info("training took %s seconds", timer.elapsed_milliseconds() / 1000)
    return None


def get_training_result(
    task_id: TaskIDType, timeout: None | float = None
) -> None | dict:
    results, _ = concurrent_context.wait_results(timeout=timeout)
    for task_id2, result in results.items():
        task_results[task_id2] |= result
    if not concurrent_context.finished(task_id):
        return None
    task_result = task_results.pop(task_id)
    log_info("finish task %s", task_id)
    stats: dict = {}
    practitioner_ids = task_result["practitioner_ids"]
    config = task_result["config"]
    assert practitioner_ids is not None
    for k, v in task_results[task_id].items():
        if k != "sv":
            stats[k] = v
            continue
        sv_dict: dict = {}
        for round_number, tmp_sv_dict in v.items():
            sv_dict[round_number] = {}
            for practitioner_id, worker_id in zip(
                sorted(practitioner_ids), range(config.worker_number), strict=False
            ):
                sv_dict[round_number][practitioner_id] = tmp_sv_dict[worker_id]
        stats[k] = sv_dict
    task_results.pop(task_id)
    return stats

import copy
import os

import torch
from cyy_naive_lib.log import add_file_handler, log_info
from cyy_naive_lib.time_counter import TimeCounter

from .config import DistributedTrainingConfig
from .context import ConcurrentFederatedLearningContext
from .task import (
    get_task_config,
    get_task_id,
    start_server,
    start_workers,
)
from .task_type import TaskIDType

# we use these environment variables to save memory in large-scale training
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["USE_THREAD_DATALOADER"] = "1"


def limit_device(device: torch.device) -> None:
    if device.type.lower() == "cuda":
        log_info("limit device %s pid %s", device, os.getpid())
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)


concurrent_context = ConcurrentFederatedLearningContext()
task_results: dict = {}


def train(
    config: DistributedTrainingConfig,
    practitioners: None | set = None,
    single_task: bool = False,
) -> TaskIDType:
    # we need to deepcopy config for concurrent training
    config = copy.deepcopy(config)
    practitioners = copy.deepcopy(practitioners)
    config.reset_session()
    config.apply_global_config()
    timer = TimeCounter()
    task_id = get_task_id()
    if practitioners is None:
        add_file_handler(config.log_file)
    task_config = get_task_config(config, practitioners=practitioners)
    start_server(task_config=task_config, single_task=single_task)
    context = start_workers(task_config=task_config, single_task=single_task)
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
    return task_id


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

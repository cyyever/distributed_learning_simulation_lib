import copy
import functools
import itertools
import os
import uuid
from collections.abc import Callable
from typing import Any

import torch
from cyy_naive_lib.log import log_debug, log_info, log_warning

from .algorithm_repository import AlgorithmRepository
from .config import DistributedTrainingConfig
from .context import (
    FederatedLearningContext,
)
from .server import AggregationServer, Server
from .task_type import TaskIDType
from .worker import Worker

type TaskConfig = dict
type TaskServerConfig = dict


def limit_device(device: torch.device) -> None:
    if device.type.lower() == "cuda":
        log_info("limit device %s pid %s", device, os.getpid())
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)


def get_task_id() -> TaskIDType:
    return uuid.uuid4()


def get_server_config(config: DistributedTrainingConfig) -> TaskServerConfig:
    assert AlgorithmRepository.has_algorithm(config.distributed_algorithm)
    context = FederatedLearningContext(worker_num=config.worker_number)
    task_id = get_task_id()
    result: dict = {"context": context, "task_id": task_id}
    result["server"] = {}
    result["server"]["constructor"] = functools.partial(
        AlgorithmRepository.create_server,
        algorithm_name=config.distributed_algorithm,
        endpoint_kwargs=config.endpoint_kwargs.get("server", {}),
        kwargs={
            "config": config,
            "task_id": task_id,
        },
    )
    return result


def get_task_config(
    config: DistributedTrainingConfig,
    practitioners: None | set = None,
) -> TaskConfig:
    result = get_server_config(config=config)
    if practitioners is None:
        practitioners = config.create_practitioners()
    else:
        config.worker_number = len(practitioners)
        for worker_id, practitioner in enumerate(
            sorted(practitioners, key=lambda p: p.id)
        ):
            assert practitioner.has_dataset(config.dc_config.dataset_name)
            practitioner.set_worker_id(worker_id)
    assert practitioners
    assert AlgorithmRepository.has_algorithm(config.distributed_algorithm)

    assert config.worker_number > 0
    assert config.round > 0
    device_allocation = config.allocate_device()
    worker_number_per_process = device_allocation["worker_number_per_process"]
    log_warning(
        "There are %s workers in total, and %s workers form a group",
        len(practitioners),
        worker_number_per_process,
    )
    process_devices = device_allocation["process_devices"]
    client_config: list[list[dict]] = []
    for batch in itertools.batched(list(practitioners), n=worker_number_per_process):
        client_config.append(
            [
                {
                    "constructor": functools.partial(
                        AlgorithmRepository.create_client,
                        algorithm_name=config.distributed_algorithm,
                        endpoint_kwargs=config.endpoint_kwargs.get("worker", {})
                        | {
                            "worker_id": practitioner.worker_id,
                        },
                        kwargs={
                            "config": config,
                            "task_id": result["task_id"],
                            "practitioner": practitioner,
                        },
                    ),
                    "device": process_devices[0] if config.preallocate_device else None,
                }
                for practitioner in batch
            ]
        )
        process_devices = process_devices[1:]
    assert client_config
    result["worker"] = client_config
    if config.preallocate_device:
        result["server"]["device"] = device_allocation["server_device"]
    return result


def get_server_impl(
    context: FederatedLearningContext, task_config: TaskConfig, **kwargs: Any
) -> Server:
    server = task_config["server"]["constructor"](context=context, **kwargs)
    log_debug("context id %d", id(context))
    return server


def start_server_impl(
    context: FederatedLearningContext, task_config: TaskConfig, **kwargs: Any
) -> dict:
    server = get_server_impl(context=context, task_config=task_config, **kwargs)
    log_debug("context id %d", id(context))

    server.start()
    log_info("stop server")

    res: dict = {}

    if isinstance(server, AggregationServer):
        res["sv"] = getattr(server.algorithm, "shapley_values", {})
        if not res["sv"]:
            res.pop("sv")
        res |= {"performance": server.performance_stat}
    return res


def start_server(
    task_config: TaskConfig, single_task: bool
) -> FederatedLearningContext:
    context = task_config.get("context")
    assert isinstance(context, FederatedLearningContext)
    device = task_config["server"].pop("device", None)
    if device is not None:
        limit_device(device)
    server_task_config = copy.copy(task_config)
    server_task_config.pop("worker")
    context.submit(
        start_server_impl, task_config=server_task_config, single_task=single_task
    )
    return context


def run_worker(constructor: Callable, **kwargs) -> None:
    worker: Worker = constructor(**kwargs)
    worker.start()


def start_workers(
    task_config: TaskConfig, single_task: bool
) -> FederatedLearningContext:
    context = task_config.get("context")
    assert isinstance(context, FederatedLearningContext)
    task_config = copy.copy(task_config)
    task_config.pop("server")
    for worker_configs in task_config["worker"]:
        for cfg in worker_configs:
            cfg["single_task"] = single_task
        log_debug(
            "run %s workers in the same process",
            len(worker_configs),
        )
        device = worker_configs[0]["device"]
        for cfg in worker_configs:
            cfg.pop("device")
        if device is not None:
            limit_device(device)
        context.submit_batch(
            funs=[run_worker] * len(worker_configs), kwargs_list=worker_configs
        )
    return context

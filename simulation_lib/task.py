import functools
import itertools
import uuid

from cyy_naive_lib.log import log_warning

from .algorithm_repository import AlgorithmRepository
from .config import DistributedTrainingConfig
from .context import FederatedLearningContext
from .server import Server
from .task_type import TaskIDType

type TaskConfig = dict
type TaskServerConfig = dict


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
    worker_number_per_process = config.get_worker_number_per_process()
    log_warning(
        "There are %s workers in total, and %s workers form a group",
        len(practitioners),
        worker_number_per_process,
    )
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
                }
                for practitioner in batch
            ]
        )
    assert client_config
    result["worker"] = client_config
    return result


def create_server(task_config: dict, **kwargs) -> Server:
    if "context" not in kwargs:
        kwargs["context"] = task_config["context"]
    return task_config["server"]["constructor"](**kwargs)

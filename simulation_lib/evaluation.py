import copy
import os

from .algorithm_repository import get_server as _get_server
from .algorithm_repository import get_task_config
from .config import DistributedTrainingConfig
from .server import AggregationServer, Server
from .task import get_task_id

# we use these environment variables to save memory in large-scale training
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["USE_THREAD_DATALOADER"] = "1"


def get_server(config: DistributedTrainingConfig) -> Server:
    # we need to deepcopy config for concurrent training
    config = copy.deepcopy(config)
    config.reset_session()
    config.apply_global_config()
    task_id = get_task_id()
    task_config = get_task_config(config, task_id=task_id)
    return _get_server(task_config=task_config)

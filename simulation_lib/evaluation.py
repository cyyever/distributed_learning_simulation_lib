import copy
import os

from .config import DistributedTrainingConfig
from .server import Server
from .task import get_server_config, get_server_impl

# we use these environment variables to save memory in large-scale training
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["USE_THREAD_DATALOADER"] = "1"


def get_server(config: DistributedTrainingConfig) -> Server:
    # we need to deepcopy config for concurrent training
    config = copy.deepcopy(config)
    task_config = get_server_config(config)
    return get_server_impl(task_config=task_config, context=task_config["context"])

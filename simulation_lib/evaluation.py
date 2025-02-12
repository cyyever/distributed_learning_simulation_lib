import copy
import os

from .algorithm_repository import get_task_config
from .config import DistributedTrainingConfig
from .context import FederatedLearningContext
from .server import AggregationServer
from .task import get_task_id

# we use these environment variables to save memory in large-scale training
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["USE_THREAD_DATALOADER"] = "1"


def get_server(
    config: DistributedTrainingConfig,
) -> AggregationServer:
    # we need to deepcopy config for concurrent training
    config = copy.deepcopy(config)
    config.reset_session()
    config.apply_global_config()
    task_id = get_task_id()
    worker_config = get_task_config(config, task_id=task_id)
    context = worker_config.pop("context")
    assert isinstance(context, FederatedLearningContext)
    server_config = worker_config.get("server", None)
    assert server_config is not None
    return server_config["constructor"](context=context, single_task=True)

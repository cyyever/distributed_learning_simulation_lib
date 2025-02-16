from cyy_naive_lib.log import set_multiprocessing_ctx
from cyy_torch_toolbox import TorchProcessContext

from .algorithm import *  # noqa: F401
from .algorithm_repository import AlgorithmRepository
from .config import *  # noqa: F401
from .evaluation import get_server
from .message import *  # noqa: F401
from .server import *  # noqa: F401
from .topology import *  # noqa: F401
from .training import train
from .util import *  # noqa: F401
from .worker import *  # noqa: F401
from .analysis import *  # noqa: F401

__all__ = ["AlgorithmRepository", "get_server", "train"]
set_multiprocessing_ctx(TorchProcessContext().get_ctx())

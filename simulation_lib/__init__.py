from cyy_naive_lib.log import set_multiprocessing_ctx
from cyy_torch_toolbox import TorchProcessContext

from .algorithm import *  # noqa: F401
from .algorithm_repository import AlgorithmRepository  # noqa: F401
from .config import *  # noqa: F401
from .message import *  # noqa: F401
from .server import *  # noqa: F401
from .topology import *  # noqa: F401
from .util import *  # noqa: F401
from .worker import *  # noqa: F401

set_multiprocessing_ctx(TorchProcessContext().get_ctx())

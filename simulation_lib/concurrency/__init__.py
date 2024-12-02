import concurrent.futures
from collections.abc import Callable, Sequence

import gevent.lock
from cyy_torch_toolbox.concurrency import TorchProcessPool


class CoroutineExcutorPool(TorchProcessPool):
    def submit_batch(self, funs: Sequence[Callable]) -> concurrent.futures.Future:
        return super().submit(self.batch_fun, funs)

    @classmethod
    def batch_fun(cls, funs, *args, **kwargs) -> None:
        assert funs
        gevent.joinall(
            [gevent.spawn(fun, *args, **kwargs) for fun in funs], raise_error=True
        )

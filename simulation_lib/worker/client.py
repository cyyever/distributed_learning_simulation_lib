from typing import Any

import gevent
import torch
from cyy_naive_lib.topology import ClientEndpoint

from ..executor import ExecutorContext
from .protocol import WorkerProtocol


class ClientMixin(WorkerProtocol):
    def _send_data_to_server(self, data: Any) -> None:
        assert isinstance(self.endpoint, ClientEndpoint)
        self.endpoint.send(data)

    def _get_data_from_server(self) -> Any:
        assert isinstance(self.endpoint, ClientEndpoint)
        self.pause()
        ExecutorContext.release()
        while not self.endpoint.has_data():
            gevent.sleep(0.1)
        ExecutorContext.acquire(self.name)
        return self.endpoint.get()

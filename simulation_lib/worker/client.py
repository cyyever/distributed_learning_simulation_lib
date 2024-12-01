from typing import Any

import gevent
from cyy_naive_lib.topology import ClientEndpoint

from .protocol import WorkerProtocol


class ClientMixin(WorkerProtocol):
    def _send_data_to_server(self, data: Any) -> None:
        assert isinstance(self.endpoint, ClientEndpoint)
        self.endpoint.send(data)

    def _get_data_from_server(self, in_round: bool = False) -> Any:
        assert isinstance(self.endpoint, ClientEndpoint)
        self.pause(in_round=in_round)
        self.device_context.release()
        while not self.endpoint.has_data():
            gevent.sleep(0.1)
        self.device_context.acquire(self.name)
        return self.endpoint.get()

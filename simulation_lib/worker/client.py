from typing import Any

from cyy_naive_lib.log import log_debug

from ..context import ClientEndpointInCoroutine
from .protocol import WorkerProtocol


class ClientMixin(WorkerProtocol):
    def _send_data_to_server(self, data: Any) -> None:
        assert isinstance(self.endpoint, ClientEndpointInCoroutine)
        log_debug("send data %s", type(data))
        self.endpoint.send(data)

    def _get_data_from_server(self, in_round: bool = False) -> Any:
        assert isinstance(self.endpoint, ClientEndpointInCoroutine)
        self.pause(in_round=in_round)
        return self.endpoint.get()

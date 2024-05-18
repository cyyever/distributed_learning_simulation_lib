import asyncio
from typing import Any

import torch
from cyy_naive_lib.topology import ClientEndpoint

from ..executor import ExecutorContext
from .protocol import WorkerProtocol


class ClientMixin(WorkerProtocol):
    def send_data_to_server(self, data: Any) -> None:
        assert isinstance(self.endpoint, ClientEndpoint)
        self.endpoint.send(data)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def _get_data_from_server(self) -> Any:
        assert isinstance(self.endpoint, ClientEndpoint)
        while not self.endpoint.has_data():
            self.pause()
            ExecutorContext.release()
            await asyncio.sleep(5)
            await ExecutorContext.acquire(self.name)
        return self.endpoint.get()

import asyncio
from typing import Any

import torch

from ..executor import ExecutorContext
from .worker import Worker


class Client(Worker):
    def send_data_to_server(self, data: Any) -> None:
        self._endpoint.send(data)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def _get_data_from_server(self) -> Any:
        while not self._endpoint.has_data():
            self.trainer.wait_stream()
            self._release_device_lock()
            ExecutorContext.release()
            await asyncio.sleep(0.1)
            await ExecutorContext.acquire(self.name)
        return self._endpoint.get()

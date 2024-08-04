import random

from .protocol import AggregationServerProtocol


class RoundSelectionMixin(AggregationServerProtocol):
    def select_workers(self) -> set:
        random_client_number: int | None = self.config.algorithm_kwargs.pop(
            "random_client_number", None
        )
        if random_client_number is not None:
            return set(
                random.sample(list(range(self.worker_number)), k=random_client_number)
            )
        return set(range(self.worker_number))

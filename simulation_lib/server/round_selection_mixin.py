import random

from .protocol import AggregationServerProtocol


class RoundSelectionMixin(AggregationServerProtocol):
    selection_result: dict[int, set[int]] = {}

    def select_workers(self) -> set[int]:
        if self.round_index in self.selection_result:
            return self.selection_result[self.round_index]
        random_client_number: int | None = self.config.algorithm_kwargs.pop(
            "random_client_number", None
        )
        result: set[int] = set()
        if random_client_number is not None:
            result = set(
                random.sample(list(range(self.worker_number)), k=random_client_number)
            )
        else:
            result = set(range(self.worker_number))
        self.selection_result[self.round_index] = result
        return result

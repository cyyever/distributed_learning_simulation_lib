from typing import override

from cyy_naive_lib.topology.cs_endpoint import ClientEndpoint

from ..dp import add_dp_noise, compute_dp_sigma
from ..message import FeatureMessage


class DifferentialPrivacyEmbeddingEndpoint(ClientEndpoint):
    def __init__(self, **kwargs) -> None:
        C: float = kwargs.pop("C", 1.0)
        epsilon: float = kwargs.pop("epsilon", 4.0)
        delta: float = kwargs.pop("delta", 1e-5)

        super().__init__(**kwargs)
        self.C = C
        self.sigma = compute_dp_sigma(epsilon=epsilon, delta=delta)

    @override
    def send(self, data) -> None:
        if isinstance(data, FeatureMessage) and data.feature is not None:
            data.feature = add_dp_noise(tensor=data.feature, C=self.C, sigma=self.sigma)
        super().send(data=data)

from typing import override

from cyy_naive_lib.topology.cs_endpoint import ClientEndpoint

from ..dp import add_dp_noise, add_dp_noise_to_parameter, compute_dp_sigma
from ..message import DeltaParameterMessage, FeatureMessage, ParameterMessage


class _DifferentialPrivacyEndpointBase(ClientEndpoint):
    def __init__(self, **kwargs) -> None:
        C: float = kwargs.pop("C", 1.0)
        epsilon: float = kwargs.pop("epsilon", 4.0)
        delta: float = kwargs.pop("delta", 1e-5)

        super().__init__(**kwargs)
        self.C = C
        self.sigma = compute_dp_sigma(epsilon=epsilon, delta=delta)


class DifferentialPrivacyEmbeddingEndpoint(_DifferentialPrivacyEndpointBase):
    @override
    def send(self, data) -> None:
        if isinstance(data, FeatureMessage) and data.feature is not None:
            data.feature = add_dp_noise(tensor=data.feature, C=self.C, sigma=self.sigma)
        super().send(data=data)


class DifferentialPrivacyParameterEndpoint(_DifferentialPrivacyEndpointBase):
    @override
    def send(self, data) -> None:
        match data:
            case ParameterMessage():
                data.parameter = add_dp_noise_to_parameter(
                    data.parameter, C=self.C, sigma=self.sigma
                )
            case DeltaParameterMessage():
                data.delta_parameter = add_dp_noise_to_parameter(
                    data.delta_parameter, C=self.C, sigma=self.sigma
                )
        super().send(data=data)

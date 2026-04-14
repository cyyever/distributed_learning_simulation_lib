import math
from typing import override

import torch
from cyy_naive_lib.log import log_info, log_warning
from cyy_naive_lib.topology.cs_endpoint import ClientEndpoint

from ..dp import add_dp_noise, add_dp_noise_to_parameter, compute_dp_sigma
from ..message import DeltaParameterMessage, FeatureMessage, ParameterMessage


def _parameter_norm(parameter) -> float:
    return torch.linalg.vector_norm(
        torch.cat([v.reshape(-1).float() for v in parameter.values()])
    ).item()


def _num_params(parameter) -> int:
    return sum(v.numel() for v in parameter.values())


class _DifferentialPrivacyEndpointBase(ClientEndpoint):
    def __init__(self, **kwargs) -> None:
        C: float = kwargs.pop("C", 1.0)
        epsilon: float = kwargs.pop("epsilon", 4.0)
        delta: float = kwargs.pop("delta", 1e-5)

        super().__init__(**kwargs)
        self.C = C
        self.sigma = compute_dp_sigma(epsilon=epsilon, delta=delta)
        log_info(
            "DP endpoint init: C=%.4f, epsilon=%.4f, delta=%.2e, sigma=%.4f",
            self.C,
            epsilon,
            delta,
            self.sigma,
        )


class DifferentialPrivacyEmbeddingEndpoint(_DifferentialPrivacyEndpointBase):
    @override
    def send(self, data) -> None:
        if isinstance(data, FeatureMessage) and data.feature is not None:
            data.feature = add_dp_noise(tensor=data.feature, C=self.C, sigma=self.sigma)
        super().send(data=data)


class DifferentialPrivacyParameterEndpoint(_DifferentialPrivacyEndpointBase):
    def _log_noise_stats(
        self, label: str, parameter, pre_norm: float, post_norm: float
    ) -> None:
        n = _num_params(parameter)
        expected_noise_norm = self.sigma * self.C * math.sqrt(n)
        snr = (
            pre_norm / expected_noise_norm if expected_noise_norm > 0 else float("inf")
        )
        log_info(
            "DP endpoint%s: pre_noise_norm=%.6f, post_noise_norm=%.6f, "
            "expected_noise_norm=%.2f, SNR=%.4f, C=%.4f, sigma=%.4f, "
            "num_params=%d",
            label,
            pre_norm,
            post_norm,
            expected_noise_norm,
            snr,
            self.C,
            self.sigma,
            n,
        )
        if snr < 1.0:
            log_warning(
                "DP endpoint%s: SNR=%.4f < 1.0, noise dominates signal! "
                "Consider increasing C (currently %.4f) to at least %.2f",
                label,
                snr,
                self.C,
                pre_norm,
            )

    @override
    def send(self, data) -> None:
        match data:
            case ParameterMessage():
                pre_norm = _parameter_norm(data.parameter)
                data.parameter = add_dp_noise_to_parameter(
                    data.parameter, C=self.C, sigma=self.sigma
                )
                post_norm = _parameter_norm(data.parameter)
                self._log_noise_stats("", data.parameter, pre_norm, post_norm)
            case DeltaParameterMessage():
                pre_norm = _parameter_norm(data.delta_parameter)
                data.delta_parameter = add_dp_noise_to_parameter(
                    data.delta_parameter, C=self.C, sigma=self.sigma
                )
                post_norm = _parameter_norm(data.delta_parameter)
                self._log_noise_stats(
                    " (delta)", data.delta_parameter, pre_norm, post_norm
                )
        super().send(data=data)

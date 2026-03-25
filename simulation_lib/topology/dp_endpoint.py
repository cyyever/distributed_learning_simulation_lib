import math
from typing import override

import torch
from cyy_naive_lib.topology.cs_endpoint import ClientEndpoint

from ..message import FeatureMessage


@torch.no_grad()
def add_dp_noise(
    tensor: torch.Tensor, C: float, sigma: float
) -> torch.Tensor:
    """Apply differential privacy (clip + Gaussian noise) to vectors.

    Args:
        tensor: Tensor of shape (batch_size, ...) or (dim,).
        C: L2 norm clipping threshold.
        sigma: Noise multiplier (without C), i.e. sqrt(2*ln(1.25/delta))/epsilon.

    Returns:
        A new tensor with clipped and noised vectors, same shape as input.
    """
    original_shape = tensor.shape
    flat = tensor.reshape(-1) if tensor.dim() <= 1 else tensor.reshape(tensor.shape[0], -1)
    norms = flat.norm(dim=-1, keepdim=True)
    clipped = flat / torch.clamp(norms / C, min=1)
    result = clipped + torch.randn_like(clipped) * (sigma * C)
    return result.reshape(original_shape)


class DifferentialPrivacyEmbeddingEndpoint(ClientEndpoint):
    def __init__(self, **kwargs) -> None:
        C: float = kwargs.pop("C", 1)
        delta: float = kwargs.pop("delta")
        epsilon: float = kwargs.pop("epsilon")
        assert C > 0, f"C must be positive, got {C}"
        assert epsilon > 0, f"epsilon must be positive, got {epsilon}"
        assert 0 < delta < 1, f"delta must be in (0, 1), got {delta}"
        super().__init__(**kwargs)
        self.C = C
        self.sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    @override
    def send(self, data) -> None:
        if isinstance(data, FeatureMessage) and data.feature is not None:
            data.feature = add_dp_noise(data.feature, self.C, self.sigma)
        super().send(data=data)

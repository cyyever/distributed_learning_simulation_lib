import math

import torch
from cyy_torch_toolbox import ModelParameter


def _flatten_per_sample(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten tensor to 2D (batch, features) or 1D."""
    return tensor.reshape(-1) if tensor.dim() <= 1 else tensor.reshape(tensor.shape[0], -1)


def compute_dp_sigma(epsilon: float = 4.0, delta: float = 1e-5) -> float:
    """Compute the noise multiplier for the Gaussian mechanism.

    Args:
        epsilon: Privacy budget, must be positive.
        delta: Privacy parameter, must be in (0, 1).

    Returns:
        Noise multiplier sigma = sqrt(2*ln(1.25/delta)) / epsilon.
    """
    assert epsilon > 0, f"epsilon must be positive, got {epsilon}"
    assert 0 < delta < 1, f"delta must be in (0, 1), got {delta}"
    return math.sqrt(2 * math.log(1.25 / delta)) / epsilon


@torch.no_grad()
def compute_dp_clipping_threshold(tensor: torch.Tensor) -> float:
    """Compute the recommended clipping threshold C as the median of L2 norms.

    Per Abadi et al. 2016, using the median of gradient/feature norms
    balances the bias-variance trade-off of clipping.

    Args:
        tensor: Tensor of shape (batch_size, ...) or (dim,).

    Returns:
        Median L2 norm as the recommended C value.
    """
    flat = _flatten_per_sample(tensor)
    norms = torch.linalg.vector_norm(flat, dim=-1)
    return norms.median().item()


@torch.no_grad()
def add_dp_noise(
    tensor: torch.Tensor, C: float | None = None, sigma: float | None = None
) -> torch.Tensor:
    """Apply differential privacy (clip + Gaussian noise) to vectors.

    Args:
        tensor: Tensor of shape (batch_size, ...) or (dim,).
        C: L2 norm clipping threshold.
        sigma: Noise multiplier (without C), i.e. sqrt(2*ln(1.25/delta))/epsilon.

    Returns:
        A new tensor with clipped and noised vectors, same shape as input.
    """
    if C is None:
        C = compute_dp_clipping_threshold(tensor)
    assert C > 0, f"C must be positive, got {C}"
    if sigma is None:
        sigma = compute_dp_sigma()
    original_shape = tensor.shape
    flat = _flatten_per_sample(tensor)
    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    clipped = flat / torch.clamp(norms / C, min=1)
    result = clipped + torch.randn_like(clipped) * (sigma * C)
    return result.reshape(original_shape)


@torch.no_grad()
def add_dp_noise_to_parameter(
    parameter: ModelParameter, C: float = 1.0, sigma: float | None = None
) -> ModelParameter:
    """Apply differential privacy to model parameters.

    Flattens all parameter tensors into one vector, clips the total L2 norm
    to C, adds Gaussian noise, then restores original shapes.

    Args:
        parameter: Dict mapping parameter names to tensors.
        C: L2 norm clipping threshold for the full parameter vector.
        sigma: Noise multiplier (without C).

    Returns:
        New ModelParameter dict with clipped and noised tensors.
    """
    assert parameter, "parameter must not be empty"
    keys = list(parameter.keys())
    devices = {parameter[k].device for k in keys}
    assert len(devices) == 1, f"all parameters must be on the same device, got {devices}"
    flat = torch.cat([parameter[k].reshape(-1).to(dtype=torch.float64) for k in keys])
    noised = add_dp_noise(flat, C=C, sigma=sigma)
    result: ModelParameter = {}
    offset = 0
    for k in keys:
        numel = parameter[k].numel()
        result[k] = noised[offset : offset + numel].reshape(parameter[k].shape).to(
            dtype=parameter[k].dtype
        )
        offset += numel
    return result

import math

import torch
from cyy_torch_toolbox import ModelParameter


def compute_dp_sigma(epsilon: float = 4.0, delta: float = 1e-5) -> float:
    """Gaussian mechanism noise multiplier: sqrt(2*ln(1.25/delta)) / epsilon."""
    assert epsilon > 0 and 0 < delta < 1
    return math.sqrt(2 * math.log(1.25 / delta)) / epsilon


@torch.no_grad()
def add_dp_noise(tensor: torch.Tensor, C: float, sigma: float) -> torch.Tensor:
    """Clip L2 norm to C and add Gaussian noise N(0, (sigma*C)^2) per element.

    For 2D+ tensors, clips and noises each row (sample) independently.
    For 1D tensors, clips and noises the whole vector.
    """
    if tensor.dim() <= 1:
        flat = tensor.reshape(1, -1)
    else:
        flat = tensor.reshape(tensor.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    clipped = flat / torch.clamp(norms / C, min=1)
    return (clipped + torch.randn_like(clipped) * (sigma * C)).reshape(tensor.shape)


@torch.no_grad()
def add_dp_noise_to_parameter(
    parameter: ModelParameter, C: float, sigma: float
) -> ModelParameter:
    """Clip total L2 norm and add Gaussian noise to a parameter dict."""
    keys = list(parameter.keys())
    flat = torch.cat([parameter[k].reshape(-1).to(dtype=torch.float64) for k in keys])
    noised = add_dp_noise(flat, C=C, sigma=sigma)
    result: ModelParameter = {}
    offset = 0
    for k in keys:
        numel = parameter[k].numel()
        result[k] = (
            noised[offset : offset + numel]
            .reshape(parameter[k].shape)
            .to(dtype=parameter[k].dtype)
        )
        offset += numel
    return result


@torch.no_grad()
def dp_clip_gradients(
    gradients: list[torch.Tensor],
    C: float,
) -> None:
    """Clip total gradient L2 norm to C in-place.

    Computes the L2 norm across all gradients. If the total norm
    exceeds C, scales all gradients uniformly so the total norm equals C.
    """
    if not gradients:
        return
    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g) for g in gradients])
    )
    scale = torch.clamp(total_norm / C, min=1)
    for g in gradients:
        g.div_(scale)


@torch.no_grad()
def dp_add_noise_to_gradients(
    gradients: list[torch.Tensor],
    C: float,
    sigma: float,
    batch_size: int,
) -> None:
    """Add DP-SGD Gaussian noise to gradients in-place.

    Adds N(0, (sigma * C / batch_size)^2) per element. The division by
    batch_size accounts for the fact that the gradient has already been
    averaged over the batch by the training framework. In standard DP-SGD
    the noise sigma * C is added to the *sum* of per-sample clipped
    gradients and then divided by batch_size; dividing here is equivalent.
    """
    noise_scale = sigma * C / batch_size
    for g in gradients:
        g.add_(torch.randn_like(g) * noise_scale)

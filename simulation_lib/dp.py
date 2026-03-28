import math

import torch
from cyy_torch_toolbox import ModelParameter


def compute_dp_sigma(epsilon: float = 4.0, delta: float = 1e-5) -> float:
    """Gaussian mechanism noise multiplier: sqrt(2*ln(1.25/delta)) / epsilon."""
    assert epsilon > 0 and 0 < delta < 1
    return math.sqrt(2 * math.log(1.25 / delta)) / epsilon


@torch.no_grad()
def add_dp_noise(
    tensor: torch.Tensor, C: float, sigma: float
) -> torch.Tensor:
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
    return (clipped + torch.randn_like(clipped) * (sigma * C)).reshape(
        tensor.shape
    )


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
        result[k] = noised[offset : offset + numel].reshape(parameter[k].shape).to(
            dtype=parameter[k].dtype
        )
        offset += numel
    return result


@torch.no_grad()
def dp_clip_and_noise_gradients(
    parameters: list[torch.nn.Parameter],
    C: float,
    sigma: float,
) -> None:
    """Batch-level DP: clip total gradient norm to C, add N(0, (sigma*C)^2) noise in-place."""
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    total_norm = torch.linalg.vector_norm(
        torch.stack(
            [torch.linalg.vector_norm(p.grad) for p in params_with_grad]
        )
    )
    scale = torch.clamp(total_norm / C, min=1)
    for p in params_with_grad:
        p.grad.div_(scale)
        p.grad.add_(torch.randn_like(p.grad) * (sigma * C))

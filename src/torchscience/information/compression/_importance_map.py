"""Importance map for spatially adaptive compression."""

from __future__ import annotations

import torch
from torch import Tensor


def importance_map(
    x: Tensor,
    *,
    method: str = "gradient",
    normalize: bool = True,
    sigma: float = 1.0,
) -> Tensor:
    """Compute spatial importance map for adaptive compression.

    Importance maps identify regions that should receive more bits
    during compression. This enables spatially adaptive rate allocation.

    Parameters
    ----------
    x : Tensor
        Input image or feature map. Shape: ``(batch, channels, height, width)``.
    method : {"gradient", "variance", "entropy", "uniform"}, default="gradient"
        Method to compute importance:
        - "gradient": Based on gradient magnitude (edges).
        - "variance": Based on local variance (texture).
        - "entropy": Based on local entropy estimate (complexity).
        - "uniform": Uniform importance (baseline).
    normalize : bool, default=True
        If True, normalize to [0, 1] range.
    sigma : float, default=1.0
        Smoothing parameter for gradient/variance computation.

    Returns
    -------
    Tensor
        Importance map. Shape: ``(batch, 1, height, width)``.
        Higher values indicate more important regions.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(4, 3, 64, 64)
    >>> imp = importance_map(x, method="gradient")
    >>> imp.shape
    torch.Size([4, 1, 64, 64])

    Notes
    -----
    Importance maps are used in several ways in learned compression:

    1. **Bit allocation**: Allocate more bits to important regions.
    2. **Mask generation**: Binary masks for adaptive coding.
    3. **Loss weighting**: Weight distortion loss by importance.

    The gradient method captures edges, which often need more bits
    to reconstruct accurately. The variance method captures texture.
    The entropy method provides information-theoretic importance.

    See Also
    --------
    gain_unit : Learned importance scaling.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")

    if x.dim() != 4:
        raise ValueError(
            f"x must be 4D (batch, channels, height, width), got {x.dim()}D"
        )

    valid_methods = {"gradient", "variance", "entropy", "uniform"}
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got '{method}'"
        )

    if method == "gradient":
        imp = _gradient_importance(x)
    elif method == "variance":
        imp = _variance_importance(x, sigma)
    elif method == "entropy":
        imp = _entropy_importance(x)
    else:  # uniform
        imp = torch.ones(
            x.shape[0],
            1,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )

    if normalize:
        # Normalize to [0, 1] per sample
        batch_size = imp.shape[0]
        imp_flat = imp.view(batch_size, -1)
        min_vals = (
            imp_flat.min(dim=-1, keepdim=True)
            .values.unsqueeze(-1)
            .unsqueeze(-1)
        )
        max_vals = (
            imp_flat.max(dim=-1, keepdim=True)
            .values.unsqueeze(-1)
            .unsqueeze(-1)
        )
        imp = (imp - min_vals) / (max_vals - min_vals + 1e-8)

    return imp


def _gradient_importance(x: Tensor) -> Tensor:
    """Compute importance based on gradient magnitude."""
    # Convert to grayscale if multi-channel
    if x.shape[1] > 1:
        gray = x.mean(dim=1, keepdim=True)
    else:
        gray = x

    # Sobel filters for gradient
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    # Pad and convolve
    padded = torch.nn.functional.pad(gray, (1, 1, 1, 1), mode="replicate")
    grad_x = torch.nn.functional.conv2d(padded, sobel_x)
    grad_y = torch.nn.functional.conv2d(padded, sobel_y)

    # Gradient magnitude
    magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

    return magnitude


def _variance_importance(x: Tensor, sigma: float) -> Tensor:
    """Compute importance based on local variance."""
    # Convert to grayscale if multi-channel
    if x.shape[1] > 1:
        gray = x.mean(dim=1, keepdim=True)
    else:
        gray = x

    # Gaussian kernel for local mean
    kernel_size = max(3, int(sigma * 6) | 1)  # Ensure odd
    kernel = _gaussian_kernel(kernel_size, sigma, x.device, x.dtype)

    # Local mean
    padding = kernel_size // 2
    padded = torch.nn.functional.pad(gray, (padding,) * 4, mode="replicate")
    local_mean = torch.nn.functional.conv2d(padded, kernel)

    # Local variance = E[x^2] - E[x]^2
    padded_sq = torch.nn.functional.pad(
        gray**2, (padding,) * 4, mode="replicate"
    )
    local_mean_sq = torch.nn.functional.conv2d(padded_sq, kernel)
    local_var = local_mean_sq - local_mean**2

    # Clamp to avoid numerical issues
    local_var = torch.clamp(local_var, min=0)

    return local_var


def _entropy_importance(x: Tensor) -> Tensor:
    """Compute importance based on local entropy estimate."""
    # Convert to grayscale if multi-channel
    if x.shape[1] > 1:
        gray = x.mean(dim=1, keepdim=True)
    else:
        gray = x

    # Use gradient magnitude as proxy for entropy
    # High gradient = high local complexity = high entropy
    grad_imp = _gradient_importance(x)

    # Add variance component
    var_imp = _variance_importance(x, sigma=1.0)

    # Combine (both indicate local complexity)
    entropy_proxy = grad_imp + var_imp

    return entropy_proxy


def _gaussian_kernel(
    size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create normalized 2D Gaussian kernel."""
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = g.outer(g)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size)


def gain_unit(
    x: Tensor,
    gain: Tensor,
    *,
    mode: str = "multiplicative",
) -> Tensor:
    """Apply spatially-varying gain for importance-based coding.

    Scales features by a learned gain map, enabling the network to
    adaptively allocate capacity to different spatial regions.

    Parameters
    ----------
    x : Tensor
        Input features. Shape: ``(batch, channels, height, width)``.
    gain : Tensor
        Gain map. Shape: ``(batch, 1, height, width)`` or ``(batch, channels, height, width)``.
    mode : {"multiplicative", "additive"}, default="multiplicative"
        How to apply gain:
        - "multiplicative": x * gain
        - "additive": x + gain

    Returns
    -------
    Tensor
        Scaled features. Same shape as input.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(4, 64, 16, 16)
    >>> gain = torch.sigmoid(torch.randn(4, 1, 16, 16))  # Learned gain
    >>> y = gain_unit(x, gain)

    Notes
    -----
    Gain units are used in learned compression to:
    - Scale latent features before quantization
    - Implement spatial attention mechanisms
    - Enable importance-weighted rate allocation

    When used with importance maps, high-gain regions receive more
    effective precision after quantization.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")

    if not isinstance(gain, Tensor):
        raise TypeError(f"gain must be a Tensor, got {type(gain).__name__}")

    valid_modes = {"multiplicative", "additive"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    # Broadcast gain to match x
    if gain.shape[1] == 1 and x.shape[1] > 1:
        gain = gain.expand(-1, x.shape[1], -1, -1)

    if mode == "multiplicative":
        return x * gain
    else:  # additive
        return x + gain

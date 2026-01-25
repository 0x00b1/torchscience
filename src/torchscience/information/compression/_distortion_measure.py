"""Distortion measure operators."""

from __future__ import annotations

import torch
from torch import Tensor


def distortion_measure(
    x: Tensor,
    y: Tensor,
    *,
    metric: str = "mse",
    reduction: str = "mean",
) -> Tensor:
    """Compute distortion between original and reconstructed signals.

    Distortion measures quantify the difference between an original
    signal and its reconstruction after compression/quantization.

    Parameters
    ----------
    x : Tensor
        Original signal. Any shape.
    y : Tensor
        Reconstructed signal. Must match shape of x.
    metric : {"mse", "mae", "rmse", "psnr", "ssim_approx"}, default="mse"
        Distortion metric:
        - "mse": Mean squared error ||x - y||²
        - "mae": Mean absolute error |x - y|
        - "rmse": Root mean squared error sqrt(mse)
        - "psnr": Peak signal-to-noise ratio in dB
        - "ssim_approx": Simplified structural similarity approximation
    reduction : {"mean", "sum", "none"}, default="mean"
        Reduction mode:
        - "mean": Average over all elements
        - "sum": Sum over all elements
        - "none": No reduction, return element-wise distortion

    Returns
    -------
    Tensor
        Distortion value. Scalar if reduction is "mean" or "sum",
        otherwise same shape as input.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(10)
    >>> y = x + 0.1 * torch.randn(10)  # Noisy reconstruction
    >>> mse = distortion_measure(x, y, metric="mse")

    Notes
    -----
    For compression, lower distortion generally means better quality
    reconstruction. The choice of metric affects perceptual quality:

    - MSE/RMSE: Simple, differentiable, but may not match perception
    - MAE: More robust to outliers than MSE
    - PSNR: Logarithmic scale, commonly used in image compression
    - SSIM: Better correlates with perceived quality (approximation here)

    See Also
    --------
    rate_distortion_lagrangian : Combined rate-distortion objective.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")
    if not isinstance(y, Tensor):
        raise TypeError(f"y must be a Tensor, got {type(y).__name__}")

    if x.shape != y.shape:
        raise ValueError(f"x shape {x.shape} must match y shape {y.shape}")

    diff = x - y

    if metric == "mse":
        distortion = diff.pow(2)
    elif metric == "mae":
        distortion = diff.abs()
    elif metric == "rmse":
        # Compute MSE first, then sqrt after reduction
        distortion = diff.pow(2)
        if reduction == "mean":
            return distortion.mean().sqrt()
        elif reduction == "sum":
            return distortion.sum().sqrt()
        else:
            return distortion.sqrt()
    elif metric == "psnr":
        # PSNR = 10 * log10(max_value^2 / MSE)
        # Assuming data range [0, 1] or will use actual max
        mse = diff.pow(2).mean()
        if mse == 0:
            return torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        max_val = max(x.max().item(), 1.0)
        psnr = 10 * torch.log10(max_val**2 / mse)
        return psnr  # PSNR doesn't use reduction parameter
    elif metric == "ssim_approx":
        # Simplified SSIM approximation
        # Full SSIM requires windowed statistics; this is a global approximation
        # SSIM ≈ (2*μx*μy + c1)(2*σxy + c2) / ((μx² + μy² + c1)(σx² + σy² + c2))
        c1 = 0.01**2
        c2 = 0.03**2

        mu_x = x.mean()
        mu_y = y.mean()
        var_x = ((x - mu_x) ** 2).mean()
        var_y = ((y - mu_y) ** 2).mean()
        cov_xy = ((x - mu_x) * (y - mu_y)).mean()

        numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        ssim = numerator / denominator
        return ssim  # SSIM is already a scalar
    else:
        raise ValueError(
            f"metric must be 'mse', 'mae', 'rmse', 'psnr', or 'ssim_approx', got '{metric}'"
        )

    # Apply reduction
    if reduction == "mean":
        return distortion.mean()
    elif reduction == "sum":
        return distortion.sum()
    elif reduction == "none":
        return distortion
    else:
        raise ValueError(
            f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
        )

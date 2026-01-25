"""Perceptual loss functions for learned compression."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def perceptual_loss(
    x: Tensor,
    y: Tensor,
    *,
    method: str = "ssim",
    window_size: int = 11,
    reduction: str = "mean",
) -> Tensor:
    """Compute perceptual loss between images.

    Perceptual losses measure differences that correlate better with
    human perception than pixel-wise metrics like MSE.

    Parameters
    ----------
    x : Tensor
        Predicted/reconstructed image. Shape: ``(batch, channels, height, width)``.
    y : Tensor
        Target/original image. Same shape as x.
    method : {"ssim", "ms_ssim", "gradient", "laplacian"}, default="ssim"
        Perceptual loss type:
        - "ssim": 1 - SSIM (structural similarity).
        - "ms_ssim": 1 - Multi-scale SSIM.
        - "gradient": Gradient magnitude difference.
        - "laplacian": Laplacian pyramid loss.
    window_size : int, default=11
        Window size for SSIM computation.
    reduction : {"mean", "sum", "none"}, default="mean"
        Reduction method for the loss.

    Returns
    -------
    Tensor
        Perceptual loss value. Scalar if reduction is "mean" or "sum",
        otherwise shape ``(batch,)``.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(4, 3, 64, 64)
    >>> y = torch.randn(4, 3, 64, 64)
    >>> loss = perceptual_loss(x, y, method="ssim")

    Notes
    -----
    Perceptual losses are crucial for learned image compression as they
    encourage reconstructions that look good to humans rather than just
    minimizing pixel error.

    SSIM measures:
    - Luminance similarity
    - Contrast similarity
    - Structure similarity

    Multi-scale SSIM evaluates these at multiple resolutions, capturing
    both fine details and overall structure.

    Gradient and Laplacian losses focus on edge preservation, which is
    important for visual quality.

    See Also
    --------
    distortion_measure : Basic distortion metrics (MSE, PSNR, etc.).
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")

    if not isinstance(y, Tensor):
        raise TypeError(f"y must be a Tensor, got {type(y).__name__}")

    if x.shape != y.shape:
        raise ValueError(f"x shape {x.shape} must match y shape {y.shape}")

    if x.dim() != 4:
        raise ValueError(
            f"x must be 4D (batch, channels, height, width), got {x.dim()}D"
        )

    valid_methods = {"ssim", "ms_ssim", "gradient", "laplacian"}
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got '{method}'"
        )

    valid_reductions = {"mean", "sum", "none"}
    if reduction not in valid_reductions:
        raise ValueError(
            f"reduction must be one of {valid_reductions}, got '{reduction}'"
        )

    if method == "ssim":
        ssim_val = _ssim(x, y, window_size)
        loss = 1 - ssim_val
    elif method == "ms_ssim":
        ms_ssim_val = _ms_ssim(x, y, window_size)
        loss = 1 - ms_ssim_val
    elif method == "gradient":
        loss = _gradient_loss(x, y)
    else:  # laplacian
        loss = _laplacian_loss(x, y)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def _gaussian_window(
    size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Create 1D Gaussian window."""
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def _create_window(
    window_size: int, channels: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Create 2D Gaussian window for SSIM."""
    sigma = 1.5
    _1d_window = _gaussian_window(window_size, sigma, device, dtype)
    _2d_window = _1d_window.outer(_1d_window)
    window = _2d_window.unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def _ssim(x: Tensor, y: Tensor, window_size: int = 11) -> Tensor:
    """Compute SSIM between x and y."""
    channels = x.shape[1]
    window = _create_window(window_size, channels, x.device, x.dtype)

    padding = window_size // 2

    # Compute means
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    # Compute variances and covariance
    sigma_x_sq = (
        F.conv2d(x**2, window, padding=padding, groups=channels) - mu_x_sq
    )
    sigma_y_sq = (
        F.conv2d(y**2, window, padding=padding, groups=channels) - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy
    )

    # Constants for stability
    C1 = 0.01**2
    C2 = 0.03**2

    # SSIM formula
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )

    # Average over spatial dimensions and channels
    return ssim_map.mean(dim=[1, 2, 3])


def _ms_ssim(x: Tensor, y: Tensor, window_size: int = 11) -> Tensor:
    """Compute multi-scale SSIM."""
    weights = torch.tensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        device=x.device,
        dtype=x.dtype,
    )

    levels = len(weights)
    mcs = []

    for i in range(levels):
        ssim_val = _ssim(x, y, window_size)

        if i < levels - 1:
            # Compute contrast-structure component
            channels = x.shape[1]
            window = _create_window(window_size, channels, x.device, x.dtype)
            padding = window_size // 2

            mu_x = F.conv2d(x, window, padding=padding, groups=channels)
            mu_y = F.conv2d(y, window, padding=padding, groups=channels)

            sigma_x_sq = (
                F.conv2d(x**2, window, padding=padding, groups=channels)
                - mu_x**2
            )
            sigma_y_sq = (
                F.conv2d(y**2, window, padding=padding, groups=channels)
                - mu_y**2
            )
            sigma_xy = (
                F.conv2d(x * y, window, padding=padding, groups=channels)
                - mu_x * mu_y
            )

            C2 = 0.03**2
            cs = (2 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2)
            mcs.append(cs.mean(dim=[1, 2, 3]))

            # Downsample
            x = F.avg_pool2d(x, kernel_size=2)
            y = F.avg_pool2d(y, kernel_size=2)
        else:
            mcs.append(ssim_val)

    # Combine scales
    mcs = torch.stack(mcs, dim=1)
    ms_ssim_val = torch.prod(mcs ** weights.unsqueeze(0), dim=1)

    return ms_ssim_val


def _gradient_loss(x: Tensor, y: Tensor) -> Tensor:
    """Compute gradient magnitude loss."""
    # Sobel filters
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

    # Convert to grayscale if needed
    if x.shape[1] > 1:
        x_gray = x.mean(dim=1, keepdim=True)
        y_gray = y.mean(dim=1, keepdim=True)
    else:
        x_gray = x
        y_gray = y

    # Compute gradients
    x_padded = F.pad(x_gray, (1, 1, 1, 1), mode="replicate")
    y_padded = F.pad(y_gray, (1, 1, 1, 1), mode="replicate")

    grad_x_x = F.conv2d(x_padded, sobel_x)
    grad_x_y = F.conv2d(x_padded, sobel_y)
    grad_y_x = F.conv2d(y_padded, sobel_x)
    grad_y_y = F.conv2d(y_padded, sobel_y)

    # Gradient magnitude difference
    mag_x = torch.sqrt(grad_x_x**2 + grad_x_y**2 + 1e-8)
    mag_y = torch.sqrt(grad_y_x**2 + grad_y_y**2 + 1e-8)

    loss = ((mag_x - mag_y) ** 2).mean(dim=[1, 2, 3])

    return loss


def _laplacian_loss(x: Tensor, y: Tensor, levels: int = 4) -> Tensor:
    """Compute Laplacian pyramid loss."""
    total_loss = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    for i in range(levels):
        # Compute difference at this level
        diff = (x - y) ** 2
        # Weight by level (finer levels weighted more)
        weight = 2 ** (levels - i - 1)
        total_loss = total_loss + weight * diff.mean(dim=[1, 2, 3])

        if i < levels - 1:
            # Downsample for next level
            x = F.avg_pool2d(x, kernel_size=2)
            y = F.avg_pool2d(y, kernel_size=2)

    # Normalize by total weight
    total_weight = sum(2 ** (levels - i - 1) for i in range(levels))
    total_loss = total_loss / total_weight

    return total_loss


def rate_loss(
    likelihoods: Tensor | list[Tensor],
    *,
    reduction: str = "mean",
) -> Tensor:
    """Compute rate loss from likelihoods.

    The rate loss is the negative log-likelihood, which corresponds
    to the theoretical bit cost under the learned entropy model.

    Parameters
    ----------
    likelihoods : Tensor or list[Tensor]
        Probability likelihoods from entropy model(s). Each tensor
        should have values in (0, 1].
    reduction : {"mean", "sum", "none"}, default="mean"
        Reduction method:
        - "mean": Average bits per sample.
        - "sum": Total bits.
        - "none": Per-sample bits.

    Returns
    -------
    Tensor
        Rate in bits. Scalar if reduction is "mean" or "sum".

    Examples
    --------
    >>> import torch
    >>> likelihoods = torch.rand(4, 192, 16, 16) * 0.9 + 0.1  # (0.1, 1.0)
    >>> rate = rate_loss(likelihoods)

    Notes
    -----
    The rate loss is computed as:
        R = -sum(log2(p)) = -sum(log(p)) / log(2)

    This measures the theoretical minimum bits needed to encode the
    latent representation given the learned probability model.

    For training, this is typically combined with distortion loss:
        L = R + lambda * D

    where lambda controls the rate-distortion trade-off.
    """
    if isinstance(likelihoods, Tensor):
        likelihoods = [likelihoods]

    for i, lik in enumerate(likelihoods):
        if not isinstance(lik, Tensor):
            raise TypeError(
                f"likelihoods[{i}] must be a Tensor, got {type(lik).__name__}"
            )

    valid_reductions = {"mean", "sum", "none"}
    if reduction not in valid_reductions:
        raise ValueError(
            f"reduction must be one of {valid_reductions}, got '{reduction}'"
        )

    # Compute bits for each likelihood tensor
    log2 = torch.log(
        torch.tensor(
            2.0, device=likelihoods[0].device, dtype=likelihoods[0].dtype
        )
    )
    total_bits = torch.zeros(
        likelihoods[0].shape[0],
        device=likelihoods[0].device,
        dtype=likelihoods[0].dtype,
    )

    for lik in likelihoods:
        # Clamp to avoid log(0)
        lik = torch.clamp(lik, min=1e-10)
        bits = -torch.log(lik) / log2
        # Sum over all dimensions except batch
        batch_bits = bits.view(bits.shape[0], -1).sum(dim=1)
        total_bits = total_bits + batch_bits

    if reduction == "mean":
        return total_bits.mean()
    elif reduction == "sum":
        return total_bits.sum()
    else:
        return total_bits

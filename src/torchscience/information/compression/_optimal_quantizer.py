"""Optimal quantizer design using Lloyd-Max algorithm."""

from __future__ import annotations

import torch
from torch import Tensor


def optimal_quantizer(
    samples: Tensor,
    n_levels: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    init: str = "uniform",
) -> tuple[Tensor, Tensor, float]:
    """Design optimal scalar quantizer using Lloyd-Max algorithm.

    The Lloyd-Max algorithm iteratively optimizes quantization levels
    and decision boundaries to minimize mean squared error for a given
    input distribution (represented by samples).

    Parameters
    ----------
    samples : Tensor
        Training samples from the distribution to be quantized.
        Shape: ``(n,)`` - must be 1D.
    n_levels : int
        Number of quantization levels (codebook size).
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence tolerance for codebook change.
    init : {"uniform", "random", "kmeans++"}, default="uniform"
        Initialization method:
        - "uniform": Uniformly spaced levels between min and max.
        - "random": Random samples from the data.
        - "kmeans++": K-means++ style initialization.

    Returns
    -------
    codebook : Tensor
        Optimized quantization levels. Shape: ``(n_levels,)``.
    boundaries : Tensor
        Decision boundaries between levels. Shape: ``(n_levels - 1,)``.
        Sample x maps to level i if boundaries[i-1] < x <= boundaries[i].
    distortion : float
        Final mean squared error.

    Examples
    --------
    >>> import torch
    >>> # Design 4-level quantizer for Gaussian distribution
    >>> samples = torch.randn(10000)
    >>> codebook, boundaries, mse = optimal_quantizer(samples, n_levels=4)
    >>> len(codebook)
    4

    Notes
    -----
    The Lloyd-Max algorithm alternates between:
    1. **Centroid update**: Set each level to the centroid of samples
       mapped to it (minimizes distortion for fixed boundaries).
    2. **Boundary update**: Set boundaries to midpoints between adjacent
       levels (optimal for MSE distortion).

    For a Gaussian distribution with variance 1:
    - 2 levels: codebook ≈ [-0.798, 0.798], MSE ≈ 0.363
    - 4 levels: MSE ≈ 0.117

    This is the foundation for PDF-optimized quantization in compression.

    See Also
    --------
    scalar_quantize : Apply quantization with a codebook.
    vector_quantize : Vector quantization (k-means for vectors).
    """
    if not isinstance(samples, Tensor):
        raise TypeError(
            f"samples must be a Tensor, got {type(samples).__name__}"
        )

    if samples.dim() != 1:
        raise ValueError(
            f"samples must be 1-dimensional, got {samples.dim()}D"
        )

    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2, got {n_levels}")

    if samples.numel() < n_levels:
        raise ValueError(
            f"Need at least {n_levels} samples, got {samples.numel()}"
        )

    samples = samples.float()
    device = samples.device

    # Initialize codebook
    if init == "uniform":
        x_min, x_max = samples.min(), samples.max()
        codebook = torch.linspace(x_min, x_max, n_levels, device=device)
    elif init == "random":
        indices = torch.randperm(len(samples))[:n_levels]
        codebook = samples[indices].sort()[0]
    elif init == "kmeans++":
        codebook = _kmeans_plusplus_init(samples, n_levels)
    else:
        raise ValueError(
            f"init must be 'uniform', 'random', or 'kmeans++', got '{init}'"
        )

    # Lloyd-Max iteration
    for _ in range(max_iter):
        old_codebook = codebook.clone()

        # Compute boundaries (midpoints between levels)
        boundaries = (codebook[:-1] + codebook[1:]) / 2

        # Assign samples to nearest level
        # For each sample, find which interval it belongs to
        assignments = torch.bucketize(samples, boundaries)

        # Update centroids
        new_codebook = torch.zeros_like(codebook)
        for i in range(n_levels):
            mask = assignments == i
            if mask.sum() > 0:
                new_codebook[i] = samples[mask].mean()
            else:
                # Keep old value if no samples assigned
                new_codebook[i] = old_codebook[i]

        codebook = new_codebook

        # Check convergence
        if (codebook - old_codebook).abs().max() < tol:
            break

    # Compute final boundaries and distortion
    boundaries = (codebook[:-1] + codebook[1:]) / 2
    assignments = torch.bucketize(samples, boundaries)
    quantized = codebook[assignments]
    distortion = ((samples - quantized) ** 2).mean().item()

    return codebook, boundaries, distortion


def _kmeans_plusplus_init(samples: Tensor, k: int) -> Tensor:
    """K-means++ initialization for codebook."""
    device = samples.device
    n = len(samples)

    # First center: random sample
    centers = [samples[torch.randint(n, (1,))].item()]

    for _ in range(k - 1):
        # Compute distances to nearest center
        centers_tensor = torch.tensor(centers, device=device)
        distances = (
            (samples.unsqueeze(1) - centers_tensor.unsqueeze(0))
            .abs()
            .min(dim=1)[0]
        )

        # Sample proportional to distance squared
        probs = distances**2
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, 1)
        centers.append(samples[idx].item())

    return torch.tensor(sorted(centers), device=device)

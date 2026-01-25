"""Gaussian conditional entropy model for learned compression."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class GaussianConditional(nn.Module):
    """Gaussian conditional entropy model for learned compression.

    Models the conditional distribution P(y|z) as a Gaussian with
    parameters (mean, scale) predicted from side information z.
    This is the core of hyperprior-based learned compression.

    Parameters
    ----------
    scale_bound : float, default=0.11
        Lower bound on scale parameter for numerical stability.
    tail_mass : float, default=1e-9
        Probability mass assigned to tails.

    Examples
    --------
    >>> import torch
    >>> from torchscience.information.compression import GaussianConditional
    >>> gc = GaussianConditional()
    >>> y = torch.randn(4, 192, 16, 16)
    >>> scales = torch.abs(torch.randn(4, 192, 16, 16)) + 0.5
    >>> means = torch.randn(4, 192, 16, 16)
    >>> y_hat, likelihoods = gc(y, scales, means)

    Notes
    -----
    The Gaussian conditional computes:
    - P(y_hat | scale, mean) = CDF_N((y_hat + 0.5 - mean)/scale) - CDF_N((y_hat - 0.5 - mean)/scale)

    This is used in conjunction with a hyperprior that predicts
    scale (and optionally mean) from side information.

    Training uses additive uniform noise; inference uses hard quantization.

    See Also
    --------
    EntropyBottleneck : For unconditional entropy models.
    """

    def __init__(
        self,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
    ):
        super().__init__()
        self.scale_bound = scale_bound
        self.tail_mass = tail_mass

        # Precompute constants
        self._log_sqrt_2pi = 0.5 * math.log(2 * math.pi)

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        """Evaluate standard normal CDF.

        Uses the complementary error function for numerical stability.
        """
        half = torch.tensor(0.5, dtype=inputs.dtype, device=inputs.device)
        const = torch.tensor(
            math.sqrt(0.5), dtype=inputs.dtype, device=inputs.device
        )
        return half * torch.erfc(-const * inputs)

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Tensor | None = None
    ) -> Tensor:
        """Compute likelihood under Gaussian distribution.

        Args:
            inputs: Values to evaluate. Shape: (...).
            scales: Scale (std) parameters. Shape: (...).
            means: Mean parameters. Shape: (...). Defaults to 0.

        Returns:
            Likelihoods. Shape: (...).
        """
        # Clamp scales
        scales = torch.clamp(scales, min=self.scale_bound)

        if means is not None:
            inputs = inputs - means

        # Compute CDF at upper and lower bounds
        upper = self._standardized_cumulative((0.5 - inputs) / scales)
        lower = self._standardized_cumulative((-0.5 - inputs) / scales)

        # Note: upper and lower are swapped because we're computing
        # P(floor(y) <= Y < floor(y) + 1) which is CDF(y+0.5) - CDF(y-0.5)
        # but our standardized inputs are (bound - input) / scale
        likelihood = upper - lower

        # Handle tails
        likelihood = torch.clamp(likelihood, min=self.tail_mass)

        return likelihood

    def forward(
        self,
        y: Tensor,
        scales: Tensor,
        means: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through Gaussian conditional.

        Args:
            y: Input latent representation. Shape: (batch, channels, ...).
            scales: Scale (std) parameters from hyperprior. Same shape as y.
            means: Mean parameters from hyperprior. Same shape as y.
                   If None, means are assumed to be 0.

        Returns:
            y_hat: Quantized (eval) or noisy (train) representation.
            likelihoods: Probability of each quantized value.
        """
        if means is not None:
            y_centered = y - means
        else:
            y_centered = y

        if self.training:
            # Training: add uniform noise
            noise = torch.rand_like(y) - 0.5
            y_hat_centered = y_centered + noise
        else:
            # Evaluation: hard quantization
            y_hat_centered = torch.round(y_centered)

        if means is not None:
            y_hat = y_hat_centered + means
        else:
            y_hat = y_hat_centered

        # Compute likelihoods
        likelihoods = self._likelihood(y_hat_centered, scales)

        return y_hat, likelihoods

    def compress(
        self,
        y: Tensor,
        scales: Tensor,
        means: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compress latents to quantized values.

        Args:
            y: Input latent representation.
            scales: Scale parameters.
            means: Mean parameters (optional).

        Returns:
            y_hat: Quantized representation.
            symbols: Integer symbols for entropy coding.
        """
        if means is not None:
            y_centered = y - means
        else:
            y_centered = y

        symbols = torch.round(y_centered).to(torch.int32)

        if means is not None:
            y_hat = symbols.float() + means
        else:
            y_hat = symbols.float()

        return y_hat, symbols

    def decompress(
        self,
        symbols: Tensor,
        means: Tensor | None = None,
    ) -> Tensor:
        """Decompress symbols to latent values.

        Args:
            symbols: Integer symbols from entropy coding.
            means: Mean parameters (optional).

        Returns:
            Reconstructed latent values.
        """
        y_hat = symbols.float()
        if means is not None:
            y_hat = y_hat + means
        return y_hat


def gaussian_conditional(
    y: Tensor,
    scales: Tensor,
    means: Tensor | None = None,
    *,
    training: bool = True,
    scale_bound: float = 0.11,
    tail_mass: float = 1e-9,
) -> tuple[Tensor, Tensor]:
    """Functional Gaussian conditional entropy model.

    Parameters
    ----------
    y : Tensor
        Input latent representation. Any shape.
    scales : Tensor
        Scale (std) parameters. Same shape as y.
    means : Tensor, optional
        Mean parameters. Same shape as y.
    training : bool, default=True
        If True, add uniform noise. If False, use hard quantization.
    scale_bound : float, default=0.11
        Lower bound on scale parameter.
    tail_mass : float, default=1e-9
        Minimum probability for numerical stability.

    Returns
    -------
    y_hat : Tensor
        Quantized (if not training) or noisy (if training) representation.
    likelihoods : Tensor
        Likelihoods under Gaussian distribution.

    Examples
    --------
    >>> import torch
    >>> y = torch.randn(4, 64, 8, 8)
    >>> scales = torch.abs(torch.randn(4, 64, 8, 8)) + 0.5
    >>> y_hat, likelihoods = gaussian_conditional(y, scales)
    """
    if not isinstance(y, Tensor):
        raise TypeError(f"y must be a Tensor, got {type(y).__name__}")

    if not isinstance(scales, Tensor):
        raise TypeError(
            f"scales must be a Tensor, got {type(scales).__name__}"
        )

    if scales.shape != y.shape:
        raise ValueError(
            f"scales shape {scales.shape} must match y shape {y.shape}"
        )

    if means is not None and means.shape != y.shape:
        raise ValueError(
            f"means shape {means.shape} must match y shape {y.shape}"
        )

    # Clamp scales
    scales = torch.clamp(scales, min=scale_bound)

    if means is not None:
        y_centered = y - means
    else:
        y_centered = y

    if training:
        noise = torch.rand_like(y) - 0.5
        y_hat_centered = y_centered + noise
    else:
        y_hat_centered = torch.round(y_centered)

    if means is not None:
        y_hat = y_hat_centered + means
    else:
        y_hat = y_hat_centered

    # Compute likelihoods using Gaussian CDF
    half = 0.5
    const = math.sqrt(0.5)

    upper = half * torch.erfc(-const * (0.5 - y_hat_centered) / scales)
    lower = half * torch.erfc(-const * (-0.5 - y_hat_centered) / scales)
    likelihoods = torch.clamp(upper - lower, min=tail_mass)

    return y_hat, likelihoods

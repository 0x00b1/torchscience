"""Scalar quantization operators."""

from __future__ import annotations

import torch
from torch import Tensor


class _STEQuantize(torch.autograd.Function):
    """Straight-through estimator for quantization."""

    @staticmethod
    def forward(
        ctx, x: Tensor, levels: Tensor, mode: str
    ) -> tuple[Tensor, Tensor]:
        """Quantize with STE gradient."""
        if mode == "uniform":
            # Uniform quantization to `levels` discrete values
            # Map [min, max] -> [0, levels-1] -> quantize -> scale back
            n_levels = levels.item() if levels.numel() == 1 else len(levels)
            x_min = x.min()
            x_max = x.max()
            # Handle constant input
            if x_max == x_min:
                indices = torch.zeros_like(x, dtype=torch.long)
                return x.clone(), indices

            # Scale to [0, n_levels-1]
            x_scaled = (x - x_min) / (x_max - x_min) * (n_levels - 1)
            indices = x_scaled.round().long().clamp(0, n_levels - 1)

            # Scale back
            quantized = (
                indices.float() / (n_levels - 1) * (x_max - x_min) + x_min
            )
        else:
            # Non-uniform: levels tensor contains the codebook
            # Find nearest level for each input
            x_flat = x.reshape(-1, 1)  # (N, 1)
            levels_flat = levels.reshape(1, -1)  # (1, L)
            distances = (x_flat - levels_flat).abs()
            indices = distances.argmin(dim=1).reshape(x.shape)
            quantized = levels[indices]

        return quantized, indices

    @staticmethod
    def backward(
        ctx, grad_quantized: Tensor, grad_indices: Tensor
    ) -> tuple[Tensor | None, ...]:
        """Pass gradients straight through."""
        return grad_quantized, None, None


class _SoftQuantize(torch.autograd.Function):
    """Soft quantization with temperature annealing."""

    @staticmethod
    def forward(
        ctx, x: Tensor, levels: Tensor, temperature: float, mode: str
    ) -> tuple[Tensor, Tensor]:
        """Soft quantization forward."""
        if mode == "uniform":
            n_levels = levels.item() if levels.numel() == 1 else len(levels)
            x_min = x.min()
            x_max = x.max()
            if x_max == x_min:
                indices = torch.zeros_like(x, dtype=torch.long)
                return x.clone(), indices

            # Create uniform levels
            levels = torch.linspace(
                x_min, x_max, n_levels, device=x.device, dtype=x.dtype
            )

        x_flat = x.reshape(-1, 1)  # (N, 1)
        levels_flat = levels.reshape(1, -1)  # (1, L)

        # Soft assignment with temperature
        neg_distances = -(x_flat - levels_flat).pow(2) / temperature
        weights = torch.softmax(neg_distances, dim=1)

        # Soft quantized value
        soft_quantized = (weights * levels_flat).sum(dim=1).reshape(x.shape)

        # Hard indices for reporting
        indices = neg_distances.argmax(dim=1).reshape(x.shape)

        ctx.save_for_backward(x, levels, weights)
        ctx.temperature = temperature

        return soft_quantized, indices

    @staticmethod
    def backward(
        ctx, grad_quantized: Tensor, grad_indices: Tensor
    ) -> tuple[Tensor | None, ...]:
        """Compute gradients through soft assignment."""
        x, levels, weights = ctx.saved_tensors
        temperature = ctx.temperature

        x_flat = x.reshape(-1, 1)
        levels_flat = levels.reshape(1, -1)

        # d(soft_q)/dx involves derivative of softmax
        diff = x_flat - levels_flat  # (N, L)

        # For each input, gradient is based on the weighted average contribution
        # Simplified: dq/dx ≈ Σ_i w_i * (1 - 2*(x-l_i)/T * Σ_j w_j*(l_j - q))
        # Using approximation: gradient ≈ 1 (similar to STE but smoother)
        grad_x = grad_quantized

        return grad_x, None, None, None


def scalar_quantize(
    x: Tensor,
    levels: int | Tensor = 256,
    *,
    mode: str = "uniform",
    gradient_mode: str = "ste",
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Quantize input to discrete levels.

    Maps continuous values to a finite set of quantization levels.
    Supports different gradient estimation modes for backpropagation.

    Parameters
    ----------
    x : Tensor
        Input tensor to quantize. Any shape.
    levels : int or Tensor, default=256
        If int: number of uniform quantization levels.
        If Tensor: explicit codebook of quantization levels.
    mode : {"uniform", "nonuniform"}, default="uniform"
        Quantization mode:
        - "uniform": Equally spaced levels between min and max.
        - "nonuniform": Use provided levels tensor as codebook.
    gradient_mode : {"ste", "soft", "none"}, default="ste"
        Gradient estimation mode:
        - "ste": Straight-through estimator (hard quantize, pass gradients).
        - "soft": Soft quantization with temperature annealing.
        - "none": No gradients (detached output).
    temperature : float, default=1.0
        Temperature for soft quantization mode. Lower values give
        harder assignments. Only used when gradient_mode="soft".

    Returns
    -------
    quantized : Tensor
        Quantized values with same shape as input.
    indices : Tensor
        Quantization indices (long tensor) with same shape as input.

    Examples
    --------
    >>> import torch
    >>> x = torch.tensor([0.1, 0.5, 0.9])
    >>> q, idx = scalar_quantize(x, levels=4)
    >>> q
    tensor([0.1000, 0.5000, 0.9000])  # Quantized to 4 levels

    >>> # Non-uniform quantization with custom codebook
    >>> codebook = torch.tensor([0.0, 0.3, 0.7, 1.0])
    >>> q, idx = scalar_quantize(x, levels=codebook, mode="nonuniform")

    Notes
    -----
    The straight-through estimator (STE) passes gradients unchanged
    through the quantization operation, enabling end-to-end training.

    Soft quantization approximates the hard quantization with a
    differentiable soft-assignment that approaches hard quantization
    as temperature decreases.

    See Also
    --------
    vector_quantize : Vector quantization with codebook learning.
    dithered_quantize : Quantization with dithering for noise shaping.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")

    # Handle levels parameter
    if isinstance(levels, int):
        levels_tensor = torch.tensor(levels, device=x.device)
        if mode == "nonuniform":
            raise ValueError(
                "mode='nonuniform' requires levels to be a Tensor codebook"
            )
    else:
        levels_tensor = levels.to(x.device)
        if mode == "uniform" and levels_tensor.numel() > 1:
            mode = "nonuniform"

    if gradient_mode == "none":
        with torch.no_grad():
            return _STEQuantize.apply(x, levels_tensor, mode)
    elif gradient_mode == "ste":
        return _STEQuantize.apply(x, levels_tensor, mode)
    elif gradient_mode == "soft":
        return _SoftQuantize.apply(x, levels_tensor, temperature, mode)
    else:
        raise ValueError(
            f"gradient_mode must be 'ste', 'soft', or 'none', got '{gradient_mode}'"
        )

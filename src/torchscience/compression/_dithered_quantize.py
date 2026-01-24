"""Dithered quantization operators."""

from __future__ import annotations

import torch
from torch import Tensor


def dithered_quantize(
    x: Tensor,
    levels: int = 256,
    *,
    dither_type: str = "subtractive",
    noise_type: str = "uniform",
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Quantize with dithering for improved perceptual quality.

    Dithering adds controlled noise before quantization to break up
    quantization artifacts and distribute error more uniformly.
    Subtractive dithering removes the noise after quantization.

    Parameters
    ----------
    x : Tensor
        Input tensor to quantize. Any shape.
    levels : int, default=256
        Number of quantization levels.
    dither_type : {"subtractive", "additive", "none"}, default="subtractive"
        Dithering mode:
        - "subtractive": Add noise before, subtract after (best quality).
        - "additive": Add noise before quantization only.
        - "none": No dithering (standard quantization).
    noise_type : {"uniform", "triangular"}, default="uniform"
        Type of dither noise:
        - "uniform": Uniform distribution in [-0.5, 0.5) * step.
        - "triangular": Triangular PDF (sum of two uniform, lower noise floor).

    Returns
    -------
    quantized : Tensor
        Quantized values with same shape as input.
    indices : Tensor
        Quantization indices (long tensor).
    dither : Tensor or None
        The dither signal used (None if dither_type="none").
        For subtractive dithering, this was subtracted from quantized.

    Examples
    --------
    >>> import torch
    >>> x = torch.linspace(0, 1, 100)
    >>> q, idx, dither = dithered_quantize(x, levels=8)

    Notes
    -----
    Subtractive dithering is particularly useful for audio and image
    processing where it can eliminate quantization distortion patterns
    at the cost of adding low-level noise.

    The quantization error with subtractive dithering is statistically
    independent of the input signal, which is desirable for many
    applications.

    Triangular dither (TPDF) is commonly used in audio as it provides
    a noise floor that doesn't modulate with the signal.

    See Also
    --------
    scalar_quantize : Basic scalar quantization.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")

    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")

    # Compute quantization step size
    x_min = x.min()
    x_max = x.max()

    # Handle constant input
    if x_max == x_min:
        indices = torch.zeros_like(x, dtype=torch.long)
        return x.clone(), indices, None

    step = (x_max - x_min) / (levels - 1)

    # Generate dither noise
    if dither_type == "none":
        dither = None
        x_dithered = x
    else:
        if noise_type == "uniform":
            # Uniform in [-0.5, 0.5) * step
            dither = (torch.rand_like(x) - 0.5) * step
        elif noise_type == "triangular":
            # Triangular PDF: sum of two uniform
            # This gives a triangular distribution in [-1, 1) * step
            u1 = torch.rand_like(x) - 0.5
            u2 = torch.rand_like(x) - 0.5
            dither = (u1 + u2) * step
        else:
            raise ValueError(
                f"noise_type must be 'uniform' or 'triangular', got '{noise_type}'"
            )

        x_dithered = x + dither

    # Quantize
    x_scaled = (x_dithered - x_min) / (x_max - x_min) * (levels - 1)
    indices = x_scaled.round().long().clamp(0, levels - 1)
    quantized = indices.float() / (levels - 1) * (x_max - x_min) + x_min

    # Subtractive dithering: remove dither from quantized
    if dither_type == "subtractive" and dither is not None:
        quantized = quantized - dither

    return quantized, indices, dither

"""Transform coding operators for compression."""

from __future__ import annotations

import torch
from torch import Tensor


def transform_code(
    x: Tensor,
    transform: str = "dct",
    *,
    quantization_step: float = 1.0,
    keep_ratio: float = 1.0,
    gradient_mode: str = "ste",
) -> tuple[Tensor, Tensor, Tensor]:
    """Transform coding with quantization for compression.

    Transform coding applies a decorrelating transform (DCT, DFT, or KLT),
    quantizes the coefficients, and optionally zeros out small coefficients.
    This is the foundation of JPEG, MP3, and many other codecs.

    Parameters
    ----------
    x : Tensor
        Input tensor. The transform is applied along the last dimension.
        Shape: ``(..., n)``.
    transform : {"dct", "dft", "identity"}, default="dct"
        Transform type:
        - "dct": Discrete Cosine Transform (energy compaction, real).
        - "dft": Discrete Fourier Transform (complex output).
        - "identity": No transform (for testing or skip connections).
    quantization_step : float, default=1.0
        Quantization step size (larger = more compression, more distortion).
    keep_ratio : float, default=1.0
        Fraction of coefficients to keep (1.0 = keep all). Coefficients
        with smallest magnitude are zeroed out.
    gradient_mode : {"ste", "soft", "none"}, default="ste"
        Gradient approximation for quantization:
        - "ste": Straight-through estimator.
        - "soft": Soft quantization (use quantization noise as-is).
        - "none": No gradients through quantization.

    Returns
    -------
    coefficients : Tensor
        Quantized transform coefficients. Same shape as input.
        For "dft", this is complex-valued.
    reconstructed : Tensor
        Reconstructed signal (inverse transform of quantized coefficients).
        Same shape as input, always real.
    mask : Tensor
        Boolean mask indicating kept coefficients. Shape: ``(..., n)``.

    Examples
    --------
    >>> import torch
    >>> # Compress a signal with DCT
    >>> x = torch.randn(100, 64)
    >>> coeffs, recon, mask = transform_code(x, transform="dct", keep_ratio=0.5)
    >>> # Only half the coefficients are non-zero
    >>> (coeffs != 0).float().mean()
    tensor(0.5000)

    Notes
    -----
    Transform coding exploits the fact that many signals have energy
    concentrated in a few transform coefficients. The typical pipeline:

    1. **Transform**: Apply decorrelating transform (DCT/DFT)
    2. **Quantize**: Round coefficients to discrete levels
    3. **Threshold**: Zero out small coefficients
    4. **Encode**: Entropy code the remaining coefficients

    The DCT is preferred for images/audio because:
    - Real coefficients (simpler than DFT)
    - Better energy compaction than DFT for typical signals
    - Boundary handling matches typical signal statistics

    For learned compression, the transform can be replaced by neural
    networks while keeping the quantization and entropy coding.

    See Also
    --------
    scalar_quantize : Quantization without transform.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}")

    if x.dim() < 1:
        raise ValueError(f"x must be at least 1D, got {x.dim()}D")

    valid_transforms = {"dct", "dft", "identity"}
    if transform not in valid_transforms:
        raise ValueError(
            f"transform must be one of {valid_transforms}, got '{transform}'"
        )

    valid_modes = {"ste", "soft", "none"}
    if gradient_mode not in valid_modes:
        raise ValueError(
            f"gradient_mode must be one of {valid_modes}, got '{gradient_mode}'"
        )

    if not 0 < keep_ratio <= 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    if quantization_step <= 0:
        raise ValueError(
            f"quantization_step must be positive, got {quantization_step}"
        )

    # Apply forward transform
    if transform == "dct":
        coeffs = _dct(x)
    elif transform == "dft":
        coeffs = torch.fft.fft(x)
    else:  # identity
        coeffs = x.clone()

    # Quantize coefficients
    if transform == "dft":
        # Complex quantization: quantize real and imag separately
        real_q = _quantize(coeffs.real, quantization_step, gradient_mode)
        imag_q = _quantize(coeffs.imag, quantization_step, gradient_mode)
        quantized = torch.complex(real_q, imag_q)
    else:
        quantized = _quantize(coeffs, quantization_step, gradient_mode)

    # Apply thresholding (keep_ratio)
    if keep_ratio < 1.0:
        n = x.shape[-1]
        n_keep = max(1, int(n * keep_ratio))

        # Find threshold based on magnitude
        if transform == "dft":
            magnitudes = quantized.abs()
        else:
            magnitudes = quantized.abs()

        # Get the n_keep largest magnitudes
        threshold = magnitudes.kthvalue(
            n - n_keep + 1, dim=-1, keepdim=True
        ).values
        mask = magnitudes >= threshold
    else:
        mask = torch.ones_like(x, dtype=torch.bool)

    # Apply mask
    quantized = quantized * mask

    # Inverse transform for reconstruction
    if transform == "dct":
        reconstructed = _idct(quantized)
    elif transform == "dft":
        reconstructed = torch.fft.ifft(quantized).real
    else:  # identity
        reconstructed = quantized.clone()

    return quantized, reconstructed, mask


def _quantize(x: Tensor, step: float, mode: str) -> Tensor:
    """Quantize tensor with given step size and gradient mode."""
    scaled = x / step
    quantized = scaled.round() * step

    if mode == "ste":
        # Straight-through: forward uses quantized, backward passes through
        return x + (quantized - x).detach()
    elif mode == "none":
        return quantized.detach()
    else:  # soft
        return quantized


def _dct(x: Tensor) -> Tensor:
    """Type-II DCT along last dimension.

    Uses the definition: X[k] = sum_n x[n] * cos(pi*k*(2n+1)/(2N))
    """
    n = x.shape[-1]
    device = x.device
    dtype = x.dtype

    # Create DCT matrix
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    ns = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    dct_matrix = torch.cos(torch.pi * k * (2 * ns + 1) / (2 * n))

    # Apply transform
    return torch.matmul(x, dct_matrix.T)


def _idct(x: Tensor) -> Tensor:
    """Type-III DCT (inverse of Type-II) along last dimension.

    X[n] = x[0]/2 + sum_{k=1}^{N-1} x[k] * cos(pi*k*(2n+1)/(2N))
    """
    n = x.shape[-1]
    device = x.device
    dtype = x.dtype

    # Create IDCT matrix (transpose of DCT matrix, scaled)
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    ns = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    idct_matrix = torch.cos(torch.pi * k * (2 * ns + 1) / (2 * n))

    # Scale factor: first column by 1/N, others by 2/N
    scale = torch.ones(n, device=device, dtype=dtype) * (2 / n)
    scale[0] = 1 / n

    return torch.matmul(x * scale, idct_matrix.T)

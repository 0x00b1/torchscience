"""Inverse Fourier transform implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Padding mode mapping
_PADDING_MODES = {
    "constant": 0,
    "reflect": 1,
    "replicate": 2,
    "circular": 3,
}

# Normalization mode mapping
_NORM_MODES = {
    "backward": 0,
    "ortho": 1,
    "forward": 2,
}


def inverse_fourier_transform(
    input: Tensor,
    *,
    n: Optional[int] = None,
    dim: int = -1,
    padding_mode: Literal[
        "constant", "reflect", "replicate", "circular"
    ] = "constant",
    padding_value: float = 0.0,
    window: Optional[Tensor] = None,
    norm: Optional[Literal["forward", "backward", "ortho"]] = None,
) -> Tensor:
    r"""Compute the inverse discrete Fourier transform of a signal.

    The inverse Fourier transform is defined as:

    .. math::
        x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{2\pi i k n / N}

    (with default ``'backward'`` normalization).

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. Typically complex-valued.
    n : int, optional
        Signal length. If given, the input will either be padded or
        truncated to this length before computing the transform.
        Default: ``None`` (use input size along ``dim``).
    dim : int, optional
        The dimension along which to compute the transform.
        Default: ``-1`` (last dimension).
    padding_mode : str, optional
        Padding mode when ``n`` is larger than input size. One of:

        - ``'constant'``: Pad with ``padding_value`` (default 0).
        - ``'reflect'``: Reflect the signal at boundaries.
        - ``'replicate'``: Replicate edge values.
        - ``'circular'``: Wrap around (periodic extension).

        Default: ``'constant'``.
    padding_value : float, optional
        Fill value for ``'constant'`` padding mode. Ignored for other modes.
        Default: ``0.0``.
    window : Tensor, optional
        Window function to apply before the transform. Must be 1-D with size
        matching the (possibly padded) signal length along ``dim``.
        Default: ``None`` (no windowing).
    norm : str, optional
        Normalization mode. One of:

        - ``'backward'``: No normalization on forward, divide by n on inverse.
        - ``'ortho'``: Normalize by 1/sqrt(n) on both forward and inverse.
        - ``'forward'``: Divide by n on forward, no normalization on inverse.

        Default: ``None`` (equivalent to ``'backward'``).

    Returns
    -------
    Tensor
        The inverse Fourier transform of the input. Complex-valued.
        If ``n`` is specified and differs from the input size along ``dim``,
        the output size along ``dim`` will be ``n``.

    Examples
    --------
    Round-trip with forward transform:

    >>> x = torch.randn(100)
    >>> X = torchscience.transform.fourier_transform(x)
    >>> x_recovered = torchscience.transform.inverse_fourier_transform(X)
    >>> torch.allclose(x_recovered.real, x, atol=1e-5)
    True

    Compare with torch.fft.ifft:

    >>> X = torch.randn(64, dtype=torch.complex64)
    >>> x = inverse_fourier_transform(X)
    >>> torch.allclose(x, torch.fft.ifft(X))
    True

    Notes
    -----
    **Normalization:**

    - ``'backward'`` (default): The inverse transform is normalized by 1/n.
    - ``'ortho'``: Normalized by 1/sqrt(n), making the transform unitary.
    - ``'forward'``: The inverse transform is unnormalized.

    **Gradient Computation:**

    Gradients are computed analytically. The IFFT is a linear operator, so:

    .. math::
        \frac{\partial L}{\partial X} = \text{FFT}\left[\frac{\partial L}{\partial x}\right]

    (with appropriate normalization adjustments). Second-order gradients are
    also supported.

    See Also
    --------
    fourier_transform : The forward Fourier transform.
    torch.fft.ifft : PyTorch's IFFT implementation.
    """
    if padding_mode not in _PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {list(_PADDING_MODES.keys())}, "
            f"got '{padding_mode}'"
        )

    norm_int = _NORM_MODES.get(norm, 0) if norm is not None else 0

    return torch.ops.torchscience.inverse_fourier_transform(
        input,
        n if n is not None else -1,
        dim,
        _PADDING_MODES[padding_mode],
        padding_value,
        window,
        norm_int,
    )

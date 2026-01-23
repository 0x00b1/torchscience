"""Spectral differentiation using FFT."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.amp import custom_fwd


def _spectral_derivative_impl(
    field: Tensor,
    dim: int,
    order: int = 1,
    dx: float = 1.0,
) -> Tensor:
    """Internal implementation of spectral derivative."""
    ndim = field.ndim
    if dim < 0:
        dim = ndim + dim

    n = field.shape[dim]

    # Compute wavenumbers
    freq = torch.fft.fftfreq(n, d=dx, device=field.device, dtype=field.dtype)
    k = 2 * math.pi * freq  # Angular wavenumber

    # Reshape k for broadcasting
    shape = [1] * ndim
    shape[dim] = n
    k = k.reshape(shape)

    # FFT -> multiply by (ik)^order -> IFFT
    f_hat = torch.fft.fft(field, dim=dim)
    f_hat = f_hat * (1j * k) ** order
    result = torch.fft.ifft(f_hat, dim=dim)

    # Return real part for real input
    if not field.is_complex():
        return result.real
    return result


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def spectral_derivative(
    field: Tensor,
    dim: int,
    order: int = 1,
    dx: float = 1.0,
) -> Tensor:
    """Compute spectral derivative using FFT.

    Computes the derivative of a periodic field using spectral methods.
    Achieves machine precision for smooth periodic functions.

    Parameters
    ----------
    field : Tensor
        Input field (must have periodic boundary conditions).
    dim : int
        Dimension along which to differentiate.
    order : int, optional
        Order of derivative. Default is 1.
    dx : float, optional
        Grid spacing. Default is 1.0.

    Returns
    -------
    Tensor
        The spectral derivative of the field.

    Notes
    -----
    Uses the property that d/dx in physical space = multiplication by ik
    in Fourier space.

    Examples
    --------
    >>> n = 64
    >>> x = torch.linspace(0, 1, n + 1)[:-1]
    >>> f = torch.sin(2 * math.pi * x)
    >>> df = spectral_derivative(f, dim=0, dx=1/n)
    >>> # df is approximately 2*pi * cos(2*pi*x)
    """
    return _spectral_derivative_impl(field, dim, order, dx)

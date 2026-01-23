"""Spectral Laplacian computation using FFT."""

from __future__ import annotations

import math
from typing import Sequence, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd


def _spectral_laplacian_impl(
    field: Tensor,
    dims: Sequence[int] | None = None,
    spacing: Union[float, Sequence[float]] = 1.0,
) -> Tensor:
    """Internal implementation of spectral Laplacian.

    Computes Laplacian using: nabla^2 f = F^{-1}[-|k|^2 * F[f]]
    """
    ndim = field.ndim

    if dims is None:
        dims = list(range(ndim))
    else:
        dims = [(d if d >= 0 else ndim + d) for d in dims]

    # Handle spacing
    if isinstance(spacing, (int, float)):
        spacings = [float(spacing)] * len(dims)
    elif isinstance(spacing, Tensor):
        spacings = spacing.tolist()
    else:
        spacings = list(spacing)

    # Compute |k|^2 - sum of squared wavenumbers
    k_squared_total = None

    for i, dim in enumerate(dims):
        n = field.shape[dim]
        dx = spacings[i] if i < len(spacings) else spacings[-1]

        freq = torch.fft.fftfreq(
            n, d=dx, device=field.device, dtype=field.dtype
        )
        k = 2 * math.pi * freq  # Angular wavenumber
        k_squared = k**2

        # Reshape for broadcasting
        shape = [1] * ndim
        shape[dim] = n
        k_squared = k_squared.reshape(shape)

        if k_squared_total is None:
            k_squared_total = k_squared
        else:
            k_squared_total = k_squared_total + k_squared

    # Transform to Fourier space (multi-dimensional)
    f_hat = torch.fft.fftn(field, dim=dims)

    # Multiply by -|k|^2
    f_hat = f_hat * (-k_squared_total)

    # Transform back
    result = torch.fft.ifftn(f_hat, dim=dims)

    # Return real part for real input
    if not field.is_complex():
        return result.real

    return result


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def spectral_laplacian(
    field: Tensor,
    dims: Sequence[int] | None = None,
    spacing: Union[float, Sequence[float]] = 1.0,
) -> Tensor:
    """Compute Laplacian using FFT spectral method.

    Uses the identity: nabla^2 f = F^{-1}[-|k|^2 * F[f]]

    Best for smooth periodic functions where it achieves spectral accuracy.

    Parameters
    ----------
    field : Tensor
        Input scalar field of shape (*batch, *spatial). Must have periodic
        boundary conditions for accurate results.
    dims : sequence of int, optional
        Spatial dimensions to compute Laplacian over. Default is all dimensions.
    spacing : float or sequence of float, optional
        Grid spacing. If a float, the same spacing is used for all dimensions.
        If a sequence, it should have the same length as the number of dimensions
        being differentiated. Default is 1.0.

    Returns
    -------
    Tensor
        Laplacian nabla^2 f with same shape as input.

    Notes
    -----
    The Laplacian is computed using spectral differentiation, which achieves
    machine precision for smooth periodic functions. The spectral Laplacian
    is defined as:

    .. math::

        \\nabla^2 f = \\sum_i \\frac{\\partial^2 f}{\\partial x_i^2}

    In Fourier space, this becomes multiplication by :math:`-|k|^2`, where
    :math:`|k|^2 = k_x^2 + k_y^2 + k_z^2` is the squared wavenumber magnitude.

    Examples
    --------
    >>> import math
    >>> n = 64
    >>> x = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    >>> y = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    >>> X, Y = torch.meshgrid(x, y, indexing='ij')
    >>> f = torch.sin(X) * torch.sin(Y)  # nabla^2 f = -2*sin(x)*sin(y)
    >>> lap = spectral_laplacian(f, spacing=2 * math.pi / n)
    """
    return _spectral_laplacian_impl(field, dims, spacing)

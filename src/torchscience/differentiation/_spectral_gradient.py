"""Spectral gradient computation using FFT."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._spectral_derivative import (
    _spectral_derivative_impl,
)


def _spectral_gradient_impl(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
) -> Tensor:
    """Internal implementation of spectral gradient."""
    ndim = field.ndim

    if dim is None:
        dims = tuple(range(ndim))
    else:
        dims = dim

    if isinstance(dx, (int, float)):
        spacings = (float(dx),) * len(dims)
    else:
        spacings = tuple(dx)

    components = []
    for i, d in enumerate(dims):
        deriv = _spectral_derivative_impl(
            field, dim=d, order=1, dx=spacings[i]
        )
        components.append(deriv)

    return torch.stack(components, dim=0)


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def spectral_gradient(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
) -> Tensor:
    """Compute spectral gradient using FFT.

    Computes the gradient of a periodic scalar field using spectral methods.
    The gradient is computed by calling spectral_derivative for each dimension
    and stacking the results.

    Parameters
    ----------
    field : Tensor
        Input scalar field with periodic boundary conditions.
    dx : float or tuple of float, optional
        Grid spacing. If a float, the same spacing is used for all dimensions.
        If a tuple, it should have the same length as the number of dimensions
        being differentiated. Default is 1.0.
    dim : tuple of int, optional
        Dimensions along which to compute the gradient. Default is all dimensions.

    Returns
    -------
    Tensor
        Gradient tensor with shape (ndim, *field.shape), where ndim is the
        number of dimensions being differentiated.

    Notes
    -----
    Uses spectral differentiation which achieves machine precision for smooth
    periodic functions. The input field must satisfy periodic boundary conditions.

    Examples
    --------
    >>> field = torch.randn(32, 32)
    >>> grad = spectral_gradient(field, dx=1/32)
    >>> grad.shape
    torch.Size([2, 32, 32])

    >>> # Different spacing per dimension
    >>> grad = spectral_gradient(field, dx=(0.1, 0.2))
    >>> grad.shape
    torch.Size([2, 32, 32])
    """
    return _spectral_gradient_impl(field, dx, dim)

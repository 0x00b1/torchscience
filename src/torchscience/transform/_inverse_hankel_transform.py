"""Inverse Hankel transform implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

__all__ = ["inverse_hankel_transform"]


def inverse_hankel_transform(
    input: Tensor,
    r: Tensor,
    k: Tensor,
    *,
    dim: int = -1,
    order: float = 0.0,
    integration_method: Literal["trapezoidal", "simpson"] = "trapezoidal",
) -> Tensor:
    r"""Compute the numerical inverse Hankel transform.

    The inverse Hankel transform of order :math:`\nu` is:

    .. math::

        \mathcal{H}^{-1}_\nu\{F\}(r) = \int_0^\infty F(k) \, J_\nu(kr) \, k \, dk

    where :math:`J_\nu` is the Bessel function of the first kind of order :math:`\nu`.

    For order 0, the Hankel transform is self-reciprocal:

    .. math::

        \mathcal{H}_0^{-1} = \mathcal{H}_0

    This function computes the transform numerically using quadrature.

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`F(k)` sampled at radial frequency points :math:`k`.
    r : Tensor
        Output radial points where to evaluate the inverse transform.
    k : Tensor
        Input radial frequency points where input is sampled.
        Must be positive and sorted in ascending order.
    dim : int, optional
        Dimension along which to integrate. Default: -1.
    order : float, optional
        Order :math:`\nu` of the Hankel transform. Default: 0.0.
    integration_method : {"trapezoidal", "simpson"}, optional
        Numerical integration method. Default: "trapezoidal".

    Returns
    -------
    Tensor
        Inverse Hankel transform :math:`f(r)` evaluated at the given :math:`r` values.
        The dimension ``dim`` is replaced by the shape of ``r``.

    Examples
    --------
    Round-trip with the Hankel transform:

    >>> import torch
    >>> import torchscience.transform as T
    >>> r = torch.linspace(0.01, 20, 2000, dtype=torch.float64)
    >>> # Gaussian: f(r) = exp(-r^2)
    >>> f = torch.exp(-r**2)
    >>> k = torch.linspace(0.01, 10, 500, dtype=torch.float64)
    >>> # Forward transform
    >>> F = T.hankel_transform(f, k, r, order=0.0)
    >>> # Inverse transform
    >>> f_reconstructed = T.inverse_hankel_transform(F, r, k, order=0.0)

    Notes
    -----
    The Hankel transform of order 0 is self-reciprocal, meaning the
    forward and inverse transforms have the same form. For other orders,
    the inverse is still the same integral form but applied to the
    transformed function.

    See Also
    --------
    hankel_transform : Forward Hankel transform.
    """
    method_map = {
        "trapezoidal": 0,
        "simpson": 1,
    }

    if integration_method not in method_map:
        raise ValueError(
            f"integration_method must be one of {list(method_map.keys())}, "
            f"got {integration_method!r}"
        )

    return torch.ops.torchscience.inverse_hankel_transform(
        input, r, k, dim, order, method_map[integration_method]
    )

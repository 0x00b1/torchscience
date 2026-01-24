"""Hankel transform implementation."""

from typing import Literal

import torch
from torch import Tensor

__all__ = ["hankel_transform"]


def hankel_transform(
    input: Tensor,
    k: Tensor,
    r: Tensor,
    *,
    dim: int = -1,
    order: float = 0.0,
    integration_method: Literal["trapezoidal", "simpson"] = "trapezoidal",
) -> Tensor:
    r"""Compute the numerical Hankel transform.

    The Hankel transform of order :math:`\nu` of a function :math:`f(r)` is:

    .. math::

        \mathcal{H}_\nu\{f\}(k) = \int_0^\infty f(r) \, J_\nu(kr) \, r \, dr

    where :math:`J_\nu` is the Bessel function of the first kind of order :math:`\nu`.

    This function computes the transform numerically using quadrature.

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`f(r)` sampled at radial points :math:`r`.
    k : Tensor
        Output radial frequency values where to evaluate the transform.
    r : Tensor
        Input radial points where input is sampled.
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
        Hankel transform :math:`F(k)` evaluated at the given :math:`k` values.
        The dimension ``dim`` is replaced by the shape of ``k``.

    Examples
    --------
    The Hankel transform of order 0 of a Gaussian is another Gaussian:

    >>> import torch
    >>> import torchscience.transform as T
    >>> r = torch.linspace(0.01, 20, 2000, dtype=torch.float64)
    >>> # Gaussian: f(r) = exp(-r^2)
    >>> f = torch.exp(-r**2)
    >>> k = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    >>> F = T.hankel_transform(f, k, r, order=0.0)

    Notes
    -----
    The Hankel transform is also known as the Fourier-Bessel transform.
    It arises naturally when taking the 2D Fourier transform of a radially
    symmetric function.

    The Hankel transform of order 0 is self-reciprocal:

    .. math::

        \mathcal{H}_0\{\mathcal{H}_0\{f\}\} = f

    Common Hankel transform pairs (order 0):

    - :math:`\mathcal{H}_0\{e^{-ar^2}\}(k) = \frac{1}{2a} e^{-k^2/(4a)}`
    - :math:`\mathcal{H}_0\{e^{-ar}\}(k) = \frac{a}{(a^2 + k^2)^{3/2}}`
    - :math:`\mathcal{H}_0\{r^{-1} e^{-ar}\}(k) = \frac{1}{\sqrt{a^2 + k^2}}`
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

    return torch.ops.torchscience.hankel_transform(
        input, k, r, dim, order, method_map[integration_method]
    )

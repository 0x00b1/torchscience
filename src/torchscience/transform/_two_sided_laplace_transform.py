"""Two-sided (bilateral) Laplace transform implementation."""

from typing import Literal

import torch
from torch import Tensor

__all__ = ["two_sided_laplace_transform"]


def two_sided_laplace_transform(
    input: Tensor,
    s: Tensor,
    t: Tensor,
    *,
    dim: int = -1,
    integration_method: Literal["trapezoidal", "simpson"] = "trapezoidal",
) -> Tensor:
    r"""Compute the numerical two-sided (bilateral) Laplace transform.

    The two-sided Laplace transform of a function :math:`f(t)` is defined as:

    .. math::

        \mathcal{B}\{f\}(s) = \int_{-\infty}^{+\infty} f(t) \, e^{-st} \, dt

    This differs from the unilateral Laplace transform by integrating over
    the entire real line rather than just :math:`[0, \infty)`.

    This function computes the transform numerically using quadrature.

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`f(t)` sampled at points :math:`t`.
    s : Tensor
        Complex frequency values where to evaluate the transform.
    t : Tensor
        Time points where input is sampled.
        Can include negative values, must be sorted in ascending order.
    dim : int, optional
        Dimension along which to integrate. Default: -1.
    integration_method : {"trapezoidal", "simpson"}, optional
        Numerical integration method. Default: "trapezoidal".

    Returns
    -------
    Tensor
        Two-sided Laplace transform :math:`F(s)` evaluated at the given :math:`s` values.
        The dimension ``dim`` is replaced by the shape of ``s``.

    Examples
    --------
    The two-sided Laplace transform of a Gaussian is another Gaussian:

    >>> import torch
    >>> import torchscience.transform as T
    >>> t = torch.linspace(-10, 10, 2000, dtype=torch.float64)
    >>> # Gaussian: f(t) = exp(-t^2)
    >>> f = torch.exp(-t**2)
    >>> s = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    >>> F = T.two_sided_laplace_transform(f, s, t)

    Notes
    -----
    The two-sided Laplace transform is also known as the bilateral Laplace
    transform. It is related to the Fourier transform by:

    .. math::

        \mathcal{B}\{f\}(i\omega) = \mathcal{F}\{f\}(\omega)

    The region of convergence (ROC) for the two-sided transform is typically
    a vertical strip in the complex s-plane, rather than a half-plane as
    with the unilateral transform.

    Common two-sided Laplace transform pairs:

    - :math:`\mathcal{B}\{e^{-a|t|}\}(s) = \frac{2a}{a^2 - s^2}` for :math:`|s| < a`
    - :math:`\mathcal{B}\{e^{-t^2}\}(s) = \sqrt{\pi} e^{s^2/4}` (Gaussian)
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

    return torch.ops.torchscience.two_sided_laplace_transform(
        input, s, t, dim, method_map[integration_method]
    )

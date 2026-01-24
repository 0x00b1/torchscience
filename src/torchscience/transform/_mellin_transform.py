"""Mellin transform implementation."""

from typing import Literal

import torch
from torch import Tensor

__all__ = ["mellin_transform"]


def mellin_transform(
    input: Tensor,
    s: Tensor,
    t: Tensor,
    *,
    dim: int = -1,
    integration_method: Literal["trapezoidal", "simpson"] = "trapezoidal",
) -> Tensor:
    r"""Compute the numerical Mellin transform.

    The Mellin transform of a function :math:`f(t)` is defined as:

    .. math::

        \mathcal{M}\{f\}(s) = \int_0^\infty f(t) \, t^{s-1} \, dt

    This function computes the transform numerically using quadrature.

    Parameters
    ----------
    input : Tensor
        Input tensor :math:`f(t)` sampled at points :math:`t`.
    s : Tensor
        Complex frequency values where to evaluate the transform.
        Must have :math:`\text{Re}(s) > 0` for convergence with typical functions.
    t : Tensor
        Time/position points where input is sampled.
        Must be positive and sorted in ascending order.
    dim : int, optional
        Dimension along which to integrate. Default: -1.
    integration_method : {"trapezoidal", "simpson"}, optional
        Numerical integration method. Default: "trapezoidal".

    Returns
    -------
    Tensor
        Mellin transform :math:`F(s)` evaluated at the given :math:`s` values.
        The dimension ``dim`` is replaced by the shape of ``s``.

    Examples
    --------
    The Mellin transform of :math:`e^{-t}` is :math:`\Gamma(s)`:

    >>> import torch
    >>> import torchscience.transform as T
    >>> t = torch.linspace(0.01, 20, 2000, dtype=torch.float64)
    >>> f = torch.exp(-t)
    >>> s = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    >>> F = T.mellin_transform(f, s, t)
    >>> # Compare with Gamma(s) = [1, 1, 2]
    >>> expected = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)

    Notes
    -----
    The Mellin transform is related to the two-sided Laplace transform by
    the substitution :math:`t = e^{-x}`:

    .. math::

        \mathcal{M}\{f\}(s) = \mathcal{B}\{f(e^{-x})\}(s)

    where :math:`\mathcal{B}` denotes the two-sided Laplace transform.

    Common Mellin transform pairs:

    - :math:`\mathcal{M}\{e^{-t}\}(s) = \Gamma(s)` for :math:`\text{Re}(s) > 0`
    - :math:`\mathcal{M}\{t^a e^{-t}\}(s) = \Gamma(s+a)` for :math:`\text{Re}(s+a) > 0`
    - :math:`\mathcal{M}\{(1+t)^{-a}\}(s) = \frac{\Gamma(s)\Gamma(a-s)}{\Gamma(a)}` for :math:`0 < \text{Re}(s) < \text{Re}(a)`
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

    return torch.ops.torchscience.mellin_transform(
        input, s, t, dim, method_map[integration_method]
    )

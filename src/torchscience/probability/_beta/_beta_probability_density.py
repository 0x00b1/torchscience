"""Beta probability density function."""

import torch
from torch import Tensor


def beta_probability_density(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    r"""Probability density function of the beta distribution.

    .. math::
        f(x; a, b) = \frac{x^{a-1} (1-x)^{b-1}}{B(a, b)}

    where :math:`B(a, b)` is the beta function.

    Parameters
    ----------
    x : Tensor
        Values in (0, 1).
    a : Tensor
        First shape parameter. Must be positive.
    b : Tensor
        Second shape parameter. Must be positive.

    Returns
    -------
    Tensor
        PDF values.
    """
    return torch.ops.torchscience.beta_probability_density(x, a, b)

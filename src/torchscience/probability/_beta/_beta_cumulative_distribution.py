"""Beta cumulative distribution function."""

import torch
from torch import Tensor


def beta_cumulative_distribution(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    r"""Cumulative distribution function of the beta distribution.

    .. math::
        F(x; a, b) = I_x(a, b)

    where :math:`I_x` is the regularized incomplete beta function.

    Parameters
    ----------
    x : Tensor
        Quantiles in [0, 1].
    a : Tensor
        First shape parameter. Must be positive.
    b : Tensor
        Second shape parameter. Must be positive.

    Returns
    -------
    Tensor
        CDF values.

    Examples
    --------
    >>> x = torch.tensor([0.25, 0.5, 0.75])
    >>> a = torch.tensor(2.0)
    >>> b = torch.tensor(5.0)
    >>> beta_cumulative_distribution(x, a, b)
    tensor([0.3672, 0.8125, 0.9727])
    """
    return torch.ops.torchscience.beta_cumulative_distribution(x, a, b)

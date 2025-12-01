"""Poisson cumulative distribution function."""

import torch
from torch import Tensor


def poisson_cumulative_distribution(k: Tensor, rate: Tensor) -> Tensor:
    r"""Cumulative distribution function of the Poisson distribution.

    .. math::
        F(k; \lambda) = Q(k+1, \lambda) = \frac{\Gamma(k+1, \lambda)}{\Gamma(k+1)}

    where Q is the upper regularized incomplete gamma function.

    Parameters
    ----------
    k : Tensor
        Number of events. Non-negative integers (floored if float).
    rate : Tensor
        Rate parameter :math:`\lambda` (mean). Must be positive.

    Returns
    -------
    Tensor
        Cumulative probability :math:`P(X \le k)`.

    Notes
    -----
    Gradients are computed with respect to rate only (k is discrete).

    Examples
    --------
    >>> k = torch.tensor([0.0, 2.0, 5.0, 10.0])
    >>> rate = torch.tensor(5.0)
    >>> poisson_cumulative_distribution(k, rate)
    tensor([0.0067, 0.1247, 0.6160, 0.9863])
    """
    return torch.ops.torchscience.poisson_cumulative_distribution(k, rate)

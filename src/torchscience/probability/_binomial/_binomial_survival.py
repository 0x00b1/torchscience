"""Binomial survival function."""

import torch
from torch import Tensor


def binomial_survival(k: Tensor, n: Tensor, p: Tensor) -> Tensor:
    r"""Survival function of the binomial distribution.

    .. math::
        S(k; n, p) = P(X > k) = I_p(k+1, n-k)

    where :math:`I` is the regularized incomplete beta function.

    Parameters
    ----------
    k : Tensor
        Number of successes. Non-negative integers (floored if float).
    n : Tensor
        Number of trials. Must be positive integer.
    p : Tensor
        Probability of success in [0, 1].

    Returns
    -------
    Tensor
        Survival probability :math:`P(X > k)`.

    Notes
    -----
    The survival function is the complement of the CDF: S(k) = 1 - F(k).
    Gradients are computed with respect to p only (k and n are discrete).

    Edge cases:
    - k < 0 returns 1.0 (all probability mass is > k)
    - k >= n returns 0.0 (no probability mass is > k)
    - p <= 0 returns 0.0 (except when k < 0)
    - p >= 1 returns 1.0 if k < n, else 0.0

    Examples
    --------
    >>> k = torch.tensor([0.0, 3.0, 5.0, 10.0])
    >>> n = torch.tensor(10.0)
    >>> p = torch.tensor(0.3)
    >>> binomial_survival(k, n, p)
    tensor([0.9718, 0.3504, 0.0473, 0.0000])
    """
    return torch.ops.torchscience.binomial_survival(k, n, p)

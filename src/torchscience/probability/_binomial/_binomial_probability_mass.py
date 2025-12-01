"""Binomial probability mass function."""

import torch
from torch import Tensor


def binomial_probability_mass(k: Tensor, n: Tensor, p: Tensor) -> Tensor:
    r"""Probability mass function of the binomial distribution.

    .. math::
        P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}

    Parameters
    ----------
    k : Tensor
        Number of successes.
    n : Tensor
        Number of trials.
    p : Tensor
        Probability of success.

    Returns
    -------
    Tensor
        Probability :math:`P(X = k)`.

    Notes
    -----
    Gradients are computed with respect to p only (k and n are discrete).

    Examples
    --------
    >>> k = torch.arange(0, 11, dtype=torch.float32)
    >>> n = torch.tensor(10.0)
    >>> p = torch.tensor(0.3)
    >>> binomial_probability_mass(k, n, p)
    tensor([0.0282, 0.1211, 0.2335, 0.2668, 0.2001, 0.1029, 0.0368, 0.0090, 0.0014, 0.0001, 0.0000])
    """
    return torch.ops.torchscience.binomial_probability_mass(k, n, p)

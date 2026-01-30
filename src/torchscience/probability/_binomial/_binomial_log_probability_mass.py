"""Binomial log probability mass function."""

import torch
from torch import Tensor


def binomial_log_probability_mass(k: Tensor, n: Tensor, p: Tensor) -> Tensor:
    r"""Log probability mass function of the binomial distribution.

    .. math::
        \log P(X = k) = \log\binom{n}{k} + k \log p + (n-k) \log(1-p)

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
        Log probability :math:`\log P(X = k)`.

    Notes
    -----
    The gradients with respect to k, n, and p are computed using:

    .. math::
        \frac{\partial}{\partial k} = -\psi(k+1) + \psi(n-k+1) + \log p - \log(1-p)

        \frac{\partial}{\partial n} = \psi(n+1) - \psi(n-k+1) + \log(1-p)

        \frac{\partial}{\partial p} = \frac{k}{p} - \frac{n-k}{1-p}

    where :math:`\psi` is the digamma function.

    Examples
    --------
    >>> k = torch.arange(0, 11, dtype=torch.float32)
    >>> n = torch.tensor(10.0)
    >>> p = torch.tensor(0.3)
    >>> binomial_log_probability_mass(k, n, p)
    tensor([-3.5666, -2.1107, -1.4544, -1.3211, -1.6087, -2.2740, -3.3020, -4.7082, -6.5576, -8.9808, -12.2061])
    """
    return torch.ops.torchscience.binomial_log_probability_mass(k, n, p)

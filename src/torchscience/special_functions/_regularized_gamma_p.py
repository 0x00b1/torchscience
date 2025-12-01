"""Regularized incomplete gamma functions."""

import torch
from torch import Tensor

__all__ = ["regularized_gamma_p"]


def regularized_gamma_p(a: Tensor, x: Tensor) -> Tensor:
    r"""Lower regularized incomplete gamma function.

    Computes P(a, x) = gamma(a, x) / Gamma(a) where gamma(a, x) is the
    lower incomplete gamma function.

    .. math::
        P(a, x) = \frac{1}{\Gamma(a)} \int_0^x t^{a-1} e^{-t} dt

    Parameters
    ----------
    a : Tensor
        Shape parameter. Must be positive.
    x : Tensor
        Integration limit. Must be non-negative.

    Returns
    -------
    Tensor
        Values of P(a, x) in range [0, 1].

    Examples
    --------
    >>> a = torch.tensor([1.0, 2.0, 3.0])
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> regularized_gamma_p(a, x)
    tensor([0.6321, 0.5940, 0.5768])

    Notes
    -----
    Special cases:

    - P(a, 0) = 0 for all a > 0
    - P(a, inf) = 1 for all a > 0
    - P(1, x) = 1 - exp(-x) (exponential CDF)

    The function is used to compute:

    - Chi-squared CDF: chi2_cumulative_distribution(x, k) = P(k/2, x/2)
    - Gamma CDF: gamma_cdf(x, k, theta) = P(k, x/theta)
    - Poisson CDF (via complement): poisson_cdf(k, lambda) = Q(k+1, lambda)

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions, Chapter 8
           https://dlmf.nist.gov/8

    See Also
    --------
    regularized_gamma_q : Upper regularized incomplete gamma Q(a, x) = 1 - P(a, x)
    """
    return torch.ops.torchscience.regularized_gamma_p(a, x)

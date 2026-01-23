"""Convert Bernoulli polynomial series to standard polynomial."""

import math

import torch

from torchscience.combinatorics._bernoulli_number import (
    _bernoulli_number_exact,
)

from ._bernoulli_polynomial_b import BernoulliPolynomialB


def bernoulli_polynomial_b_to_polynomial(a: BernoulliPolynomialB):
    """Convert Bernoulli polynomial series to standard polynomial.

    Given f(x) = sum_{k=0}^{n} c[k] * B_k(x), converts to standard
    polynomial representation f(x) = sum_{j=0}^{m} d[j] * x^j.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Bernoulli polynomial series.

    Returns
    -------
    Polynomial
        Standard polynomial representation.

    Notes
    -----
    Uses the expansion B_n(x) = sum_{k=0}^{n} C(n,k) * B_k * x^{n-k}
    where B_k are Bernoulli numbers.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([0.0, 1.0]))  # B_1(x) = x - 1/2
    >>> p = bernoulli_polynomial_b_to_polynomial(a)
    >>> p  # Should be approximately [-0.5, 1.0] (constant, linear)
    """
    from torchscience.polynomial._polynomial import Polynomial

    coeffs = a.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    if n == 0:
        return Polynomial(
            torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    # The degree of the result is at most n-1 (max degree in Bernoulli basis)
    max_degree = n - 1

    # Initialize result coefficient tensor
    batch_shape = coeffs.shape[:-1]
    result_shape = batch_shape + (max_degree + 1,)
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # For each Bernoulli polynomial B_k(x) with coefficient c[k]:
    # B_k(x) = sum_{j=0}^{k} C(k,j) * B_j * x^{k-j}
    for k in range(n):
        c_k = coeffs[..., k]  # Coefficient of B_k(x)
        # Expand B_k(x)
        for j in range(k + 1):
            binom = math.comb(k, j)
            bernoulli_j = float(_bernoulli_number_exact(j))
            power = k - j  # Power of x
            result[..., power] = result[..., power] + c_k * binom * bernoulli_j

    return Polynomial(result)

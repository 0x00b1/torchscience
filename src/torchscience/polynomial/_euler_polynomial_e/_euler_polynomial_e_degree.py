"""Degree of Euler polynomial series."""

import torch

from ._euler_polynomial_e import EulerPolynomialE


def euler_polynomial_e_degree(a: EulerPolynomialE) -> int:
    """Return the degree of an Euler polynomial series.

    The degree is the highest index k with non-zero coefficient c[k].

    Parameters
    ----------
    a : EulerPolynomialE
        Euler polynomial series.

    Returns
    -------
    int
        Degree of the series.
    """
    coeffs = a.as_subclass(torch.Tensor)

    # Find highest non-zero coefficient
    # For batched input, we consider the maximum degree across all batches
    n = coeffs.shape[-1]

    for k in range(n - 1, -1, -1):
        if torch.any(coeffs[..., k] != 0):
            return k

    return 0

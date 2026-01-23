"""Convert Euler polynomial series to standard polynomial."""

import math

import torch

from torchscience.combinatorics._euler_number import _euler_number_exact

from ._euler_polynomial_e import EulerPolynomialE


def euler_polynomial_e_to_polynomial(a: EulerPolynomialE):
    """Convert Euler polynomial series to standard polynomial.

    Given f(x) = sum_{k=0}^{n} c[k] * E_k(x), converts to standard
    polynomial representation f(x) = sum_{j=0}^{m} d[j] * x^j.

    Parameters
    ----------
    a : EulerPolynomialE
        Euler polynomial series.

    Returns
    -------
    Polynomial
        Standard polynomial representation.

    Notes
    -----
    Uses the expansion E_n(x) = sum_{k=0}^{n} C(n,k) * E_k / 2^k * (x - 1/2)^{n-k}
    where E_k are Euler numbers.
    """
    from torchscience.polynomial._polynomial import Polynomial

    coeffs = a.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    if n == 0:
        return Polynomial(
            torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    # The degree of the result is at most n-1 (max degree in Euler basis)
    max_degree = n - 1

    # Initialize result coefficient tensor
    batch_shape = coeffs.shape[:-1]
    result_shape = batch_shape + (max_degree + 1,)
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # For each Euler polynomial E_k(x) with coefficient c[k]:
    # E_k(x) = sum_{j=0}^{k} C(k,j) * E_j / 2^j * (x - 1/2)^{k-j}
    # We need to expand (x - 1/2)^{k-j} = sum_{m=0}^{k-j} C(k-j,m) * x^m * (-1/2)^{k-j-m}
    for k in range(n):
        c_k = coeffs[..., k]  # Coefficient of E_k(x)
        # Expand E_k(x)
        for j in range(k + 1):
            binom_kj = math.comb(k, j)
            euler_j = float(_euler_number_exact(j))
            coeff_j = binom_kj * euler_j / (2**j)

            # Expand (x - 1/2)^{k-j}
            power = k - j
            for m in range(power + 1):
                binom_pm = math.comb(power, m)
                x_power = m
                constant_part = (-0.5) ** (power - m)
                result[..., x_power] = (
                    result[..., x_power]
                    + c_k * coeff_j * binom_pm * constant_part
                )

    return Polynomial(result)

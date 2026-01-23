"""Convert standard polynomial to Euler polynomial series."""

import math

import torch

from torchscience.combinatorics._euler_number import _euler_number_exact

from ._euler_polynomial_e import (
    EulerPolynomialE,
    euler_polynomial_e,
)


def polynomial_to_euler_polynomial_e(p) -> EulerPolynomialE:
    """Convert standard polynomial to Euler polynomial series.

    Given f(x) = sum_{j=0}^{m} a[j] * x^j, converts to Euler polynomial
    representation f(x) = sum_{k=0}^{n} c[k] * E_k(x).

    Parameters
    ----------
    p : Polynomial
        Standard polynomial.

    Returns
    -------
    EulerPolynomialE
        Euler polynomial series representation.

    Notes
    -----
    The conversion is done by solving a triangular system.
    Since E_k(x) has degree k and leading coefficient 1, we can
    work from highest degree down.
    """

    coeffs = p.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]  # n = degree + 1

    if n == 0:
        return euler_polynomial_e(
            torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    batch_shape = coeffs.shape[:-1]

    # Build the transformation matrix M where:
    # [x^0, x^1, ..., x^n]^T = M @ [E_0, E_1, ..., E_n]^T
    # M[i,j] = coefficient of x^i in E_j(x)

    M = torch.zeros(n, n, dtype=torch.float64, device=coeffs.device)
    for j in range(n):  # Column index (Euler polynomial index)
        # E_j(x) = sum_{k=0}^{j} C(j,k) * E_k / 2^k * (x - 1/2)^{j-k}
        for k in range(j + 1):
            binom_jk = math.comb(j, k)
            euler_k = float(_euler_number_exact(k))
            coeff_k = binom_jk * euler_k / (2**k)

            # Expand (x - 1/2)^{j-k}
            power = j - k
            for m in range(power + 1):
                binom_pm = math.comb(power, m)
                x_power = m  # Row index
                constant_part = (-0.5) ** (power - m)
                M[x_power, j] = (
                    M[x_power, j] + coeff_k * binom_pm * constant_part
                )

    # M is upper triangular. We need to solve M @ c = a for c
    # where a are the standard polynomial coefficients
    # c = M^{-1} @ a

    # Convert coeffs to float64 for numerical stability
    coeffs_f64 = coeffs.to(torch.float64)

    # Solve the system using triangular solve
    # M is upper triangular, so we use torch.linalg.solve_triangular
    if len(batch_shape) == 0:
        # No batch dimensions - need to add column dimension for solve_triangular
        result = torch.linalg.solve_triangular(
            M, coeffs_f64.unsqueeze(-1), upper=True
        ).squeeze(-1)
    else:
        # Has batch dimensions - flatten, solve, reshape
        B = coeffs_f64[..., 0].numel()
        coeffs_flat = coeffs_f64.reshape(B, n)
        result_flat = torch.linalg.solve_triangular(
            M.unsqueeze(0).expand(B, n, n),
            coeffs_flat.unsqueeze(-1),
            upper=True,
        ).squeeze(-1)
        result = result_flat.reshape(*batch_shape, n)

    # Convert back to original dtype
    result = result.to(coeffs.dtype)

    return euler_polynomial_e(result)

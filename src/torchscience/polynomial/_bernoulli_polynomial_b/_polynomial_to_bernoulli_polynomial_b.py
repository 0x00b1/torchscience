"""Convert standard polynomial to Bernoulli polynomial series."""

import math

import torch

from torchscience.combinatorics._bernoulli_number import (
    _bernoulli_number_exact,
)

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
    bernoulli_polynomial_b,
)


def polynomial_to_bernoulli_polynomial_b(p) -> BernoulliPolynomialB:
    """Convert standard polynomial to Bernoulli polynomial series.

    Given f(x) = sum_{j=0}^{m} a[j] * x^j, converts to Bernoulli polynomial
    representation f(x) = sum_{k=0}^{n} c[k] * B_k(x).

    Parameters
    ----------
    p : Polynomial
        Standard polynomial.

    Returns
    -------
    BernoulliPolynomialB
        Bernoulli polynomial series representation.

    Notes
    -----
    The conversion is done by solving a triangular system.
    Since B_k(x) has degree k and leading coefficient 1, we can
    work from highest degree down.

    For a polynomial of degree n, we express:
    x^n = B_n(x) + lower order terms

    More precisely, x^n in terms of Bernoulli polynomials uses:
    x^n = sum_{k=0}^{n} S(n,k) * B_k(x)
    where S(n,k) are coefficients derived from the inverse relationship.

    Examples
    --------
    >>> from torchscience.polynomial import polynomial
    >>> p = polynomial(torch.tensor([-0.5, 1.0]))  # -1/2 + x = B_1(x)
    >>> b = polynomial_to_bernoulli_polynomial_b(p)
    >>> b  # Should be [0.0, 1.0]
    """

    coeffs = p.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]  # n = degree + 1

    if n == 0:
        return bernoulli_polynomial_b(
            torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    batch_shape = coeffs.shape[:-1]

    # Build the transformation matrix M where:
    # [x^0, x^1, ..., x^n]^T = M @ [B_0, B_1, ..., B_n]^T
    # M[i,j] = coefficient of x^i in B_j(x)
    # B_j(x) = sum_{k=0}^{j} C(j,k) * B_k * x^{j-k}
    # So M[i,j] = C(j, j-i) * B_{j-i} if i <= j, else 0

    M = torch.zeros(n, n, dtype=torch.float64, device=coeffs.device)
    for j in range(n):  # Column index (Bernoulli polynomial index)
        for i in range(j + 1):  # Row index (power of x)
            k = j - i  # Index into Bernoulli number
            binom = math.comb(j, k)
            bernoulli_k = float(_bernoulli_number_exact(k))
            M[i, j] = binom * bernoulli_k

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

    return bernoulli_polynomial_b(result)

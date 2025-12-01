import torch

from .._polynomial import Polynomial
from ._jacobi_polynomial_p import JacobiPolynomialP


def jacobi_polynomial_p_to_polynomial(
    c: JacobiPolynomialP,
) -> Polynomial:
    """Convert Jacobi series to power polynomial.

    Parameters
    ----------
    c : JacobiPolynomialP
        Jacobi series.

    Returns
    -------
    Polynomial
        Equivalent power polynomial.

    Notes
    -----
    Uses the recurrence relation:
        P_0^{(α,β)}(x) = 1
        P_1^{(α,β)}(x) = (α - β)/2 + (α + β + 2)/2 * x

        For n >= 1:
        a_n = 2(n+1)(n+α+β+1)(2n+α+β)
        b_n = (2n+α+β+1)(α²-β²)
        c_n = (2n+α+β)(2n+α+β+1)(2n+α+β+2)
        d_n = 2(n+α)(n+β)(2n+α+β+2)

        P_{n+1}^{(α,β)}(x) = ((b_n + c_n*x) * P_n - d_n * P_{n-1}) / a_n

    to build power representations of each P_k^{(α,β)}, then combines them.

    Examples
    --------
    >>> c = jacobi_polynomial_p(torch.tensor([0.0, 0.0, 1.0]), alpha=0.0, beta=0.0)
    >>> p = jacobi_polynomial_p_to_polynomial(c)
    >>> p.coeffs  # P_2^{(0,0)} = (3x^2 - 1)/2 (Legendre)
    tensor([-0.5,  0.0,  1.5])
    """
    coeffs = c.coeffs
    alpha = c.alpha
    beta = c.beta
    ab = alpha + beta
    n = coeffs.shape[-1]

    # Build power representations of P_k^{(α,β)} for k = 0, ..., n-1
    # P_k_power[j] is the coefficient of x^j in P_k^{(α,β)}
    P_power = []

    # P_0^{(α,β)}(x) = 1
    P_power.append(
        torch.tensor([1.0], dtype=coeffs.dtype, device=coeffs.device)
    )

    if n > 1:
        # P_1^{(α,β)}(x) = (α - β)/2 + (α + β + 2)/2 * x
        c0 = (alpha - beta) / 2
        c1 = (ab + 2) / 2
        P_power.append(
            torch.tensor(
                [c0.item(), c1.item()],
                dtype=coeffs.dtype,
                device=coeffs.device,
            )
        )

    # Recurrence for higher degrees
    for k in range(1, n - 1):
        k_f = float(k)
        two_k_ab = 2 * k_f + ab

        # Recurrence coefficients
        a_k = 2 * (k_f + 1) * (k_f + ab + 1) * two_k_ab
        b_k = (two_k_ab + 1) * (alpha * alpha - beta * beta)
        c_k = two_k_ab * (two_k_ab + 1) * (two_k_ab + 2)
        d_k = 2 * (k_f + alpha) * (k_f + beta) * (two_k_ab + 2)

        P_k = P_power[k]
        P_km1 = P_power[k - 1]

        # P_{k+1} = ((b_k + c_k*x) * P_k - d_k * P_{k-1}) / a_k
        # = (b_k * P_k + c_k * x * P_k - d_k * P_{k-1}) / a_k

        # b_k * P_k
        term1 = b_k * P_k

        # c_k * x * P_k: shift coefficients
        P_k_shifted = torch.zeros(
            len(P_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        P_k_shifted[1:] = c_k * P_k

        # d_k * P_{k-1}
        P_km1_padded = torch.zeros(
            len(P_k_shifted), dtype=coeffs.dtype, device=coeffs.device
        )
        P_km1_padded[: len(P_km1)] = d_k * P_km1

        # Pad term1 to same length as P_k_shifted
        term1_padded = torch.zeros(
            len(P_k_shifted), dtype=coeffs.dtype, device=coeffs.device
        )
        term1_padded[: len(term1)] = term1

        P_kp1 = (term1_padded + P_k_shifted - P_km1_padded) / a_k
        P_power.append(P_kp1)

    # Combine: p = sum(c_k * P_k^{(α,β)})
    # Result has degree n-1
    result = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)

    for k in range(n):
        c_k = coeffs[..., k]
        P_k = P_power[k]
        result[: len(P_k)] = result[: len(P_k)] + c_k * P_k

    return Polynomial(coeffs=result)

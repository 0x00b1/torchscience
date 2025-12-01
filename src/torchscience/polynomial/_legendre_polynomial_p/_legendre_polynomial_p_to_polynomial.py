import torch

from torchscience.polynomial._polynomial import Polynomial

from ._legendre_polynomial_p import LegendrePolynomialP


def legendre_polynomial_p_to_polynomial(
    c: LegendrePolynomialP,
) -> Polynomial:
    """Convert Legendre series to power polynomial.

    Parameters
    ----------
    c : LegendrePolynomialP
        Legendre series.

    Returns
    -------
    Polynomial
        Equivalent power polynomial.

    Notes
    -----
    Uses the recurrence relation:
        P_0(x) = 1
        P_1(x) = x
        (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)

    to build power representations of each P_k, then combines them.

    Examples
    --------
    >>> c = legendre_polynomial_p(torch.tensor([0.0, 0.0, 1.0]))  # P_2
    >>> p = legendre_polynomial_p_to_polynomial(c)
    >>> p.coeffs  # P_2 = (3x^2 - 1)/2
    tensor([-0.5,  0.0,  1.5])
    """
    coeffs = c.coeffs
    n = coeffs.shape[-1]

    # Build power representations of P_k for k = 0, ..., n-1
    # P_k_power[j] is the coefficient of x^j in P_k
    P_power = []

    # P_0 = 1
    P_power.append(
        torch.tensor([1.0], dtype=coeffs.dtype, device=coeffs.device)
    )

    if n > 1:
        # P_1 = x
        P_power.append(
            torch.tensor([0.0, 1.0], dtype=coeffs.dtype, device=coeffs.device)
        )

    # Recurrence: (k+1)*P_{k+1} = (2k+1)*x*P_k - k*P_{k-1}
    # P_{k+1} = [(2k+1)*x*P_k - k*P_{k-1}] / (k+1)
    for k in range(1, n - 1):
        P_k = P_power[k]
        P_km1 = P_power[k - 1]

        # (2k+1)*x*P_k: shift coefficients and multiply
        P_k_shifted = torch.zeros(
            len(P_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        P_k_shifted[1:] = (2 * k + 1) * P_k

        # Pad P_{k-1} to same length
        P_km1_padded = torch.zeros(
            len(P_k_shifted), dtype=coeffs.dtype, device=coeffs.device
        )
        P_km1_padded[: len(P_km1)] = k * P_km1

        P_kp1 = (P_k_shifted - P_km1_padded) / (k + 1)
        P_power.append(P_kp1)

    # Combine: p = sum(c_k * P_k)
    # Result has degree n-1
    result = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)

    for k in range(n):
        c_k = coeffs[..., k]
        P_k = P_power[k]
        result[: len(P_k)] = result[: len(P_k)] + c_k * P_k

    return Polynomial(coeffs=result)

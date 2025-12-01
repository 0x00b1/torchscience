import torch

from torchscience.polynomial._polynomial import Polynomial

from ._chebyshev_polynomial_v import ChebyshevPolynomialV


def chebyshev_polynomial_v_to_polynomial(
    c: ChebyshevPolynomialV,
) -> Polynomial:
    """Convert Chebyshev V series to power polynomial.

    Parameters
    ----------
    c : ChebyshevPolynomialV
        Chebyshev V series.

    Returns
    -------
    Polynomial
        Equivalent power polynomial.

    Notes
    -----
    Uses the recurrence relation:
        V_0(x) = 1
        V_1(x) = 2x - 1
        V_{n+1}(x) = 2*x*V_n(x) - V_{n-1}(x)

    to build power representations of each V_k, then combines them.

    Examples
    --------
    >>> c = chebyshev_polynomial_v(torch.tensor([0.0, 0.0, 1.0]))  # V_2
    >>> p = chebyshev_polynomial_v_to_polynomial(c)
    >>> p.coeffs  # V_2 = 4x^2 - 4x + 1
    tensor([ 1., -4.,  4.])
    """
    coeffs = c.coeffs
    n = coeffs.shape[-1]

    # Build power representations of V_k for k = 0, ..., n-1
    # V_k_power[j] is the coefficient of x^j in V_k
    V_power = []

    # V_0 = 1
    V_power.append(
        torch.tensor([1.0], dtype=coeffs.dtype, device=coeffs.device)
    )

    if n > 1:
        # V_1 = 2x - 1
        V_power.append(
            torch.tensor([-1.0, 2.0], dtype=coeffs.dtype, device=coeffs.device)
        )

    # Recurrence: V_{k+1} = 2*x*V_k - V_{k-1}
    for k in range(1, n - 1):
        V_k = V_power[k]
        V_km1 = V_power[k - 1]

        # 2*x*V_k: shift coefficients and multiply by 2
        V_k_shifted = torch.zeros(
            len(V_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        V_k_shifted[1:] = 2.0 * V_k

        # Pad V_{k-1} to same length
        V_km1_padded = torch.zeros(
            len(V_k_shifted), dtype=coeffs.dtype, device=coeffs.device
        )
        V_km1_padded[: len(V_km1)] = V_km1

        V_kp1 = V_k_shifted - V_km1_padded
        V_power.append(V_kp1)

    # Combine: p = sum(c_k * V_k)
    # Result has degree n-1
    result = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)

    for k in range(n):
        c_k = coeffs[..., k]
        V_k = V_power[k]
        result[: len(V_k)] = result[: len(V_k)] + c_k * V_k

    return Polynomial(coeffs=result)

import torch

from torchscience.polynomial._polynomial import Polynomial, polynomial

from ._chebyshev_polynomial_t import ChebyshevPolynomialT


def chebyshev_polynomial_t_to_polynomial(
    c: ChebyshevPolynomialT,
) -> Polynomial:
    """Convert Chebyshev series to power polynomial.

    Parameters
    ----------
    c : ChebyshevPolynomialT
        Chebyshev series.

    Returns
    -------
    Polynomial
        Equivalent power polynomial.

    Notes
    -----
    Uses the recurrence relation:
        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)

    to build power representations of each T_k, then combines them.

    Examples
    --------
    >>> c = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
    >>> p = chebyshev_polynomial_t_to_polynomial(c)
    >>> p  # T_2 = 2x^2 - 1
    Polynomial(tensor([-1.,  0.,  2.]))
    """
    # The polynomial IS the coefficients tensor
    coeffs = c.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Build power representations of T_k for k = 0, ..., n-1
    # T_k_power[j] is the coefficient of x^j in T_k
    T_power = []

    # T_0 = 1
    T_power.append(
        torch.tensor([1.0], dtype=coeffs.dtype, device=coeffs.device)
    )

    if n > 1:
        # T_1 = x
        T_power.append(
            torch.tensor([0.0, 1.0], dtype=coeffs.dtype, device=coeffs.device)
        )

    # Recurrence: T_{k+1} = 2*x*T_k - T_{k-1}
    for k in range(1, n - 1):
        T_k = T_power[k]
        T_km1 = T_power[k - 1]

        # 2*x*T_k: shift coefficients and multiply by 2
        T_k_shifted = torch.zeros(
            len(T_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        T_k_shifted[1:] = 2.0 * T_k

        # Pad T_{k-1} to same length
        T_km1_padded = torch.zeros(
            len(T_k_shifted), dtype=coeffs.dtype, device=coeffs.device
        )
        T_km1_padded[: len(T_km1)] = T_km1

        T_kp1 = T_k_shifted - T_km1_padded
        T_power.append(T_kp1)

    # Combine: p = sum(c_k * T_k)
    # Result has degree n-1
    result = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)

    for k in range(n):
        c_k = coeffs[..., k]
        T_k = T_power[k]
        result[: len(T_k)] = result[: len(T_k)] + c_k * T_k

    return polynomial(result)

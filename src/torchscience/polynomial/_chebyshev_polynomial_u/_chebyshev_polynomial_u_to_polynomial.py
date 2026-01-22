import torch

from torchscience.polynomial._polynomial import Polynomial, polynomial

from ._chebyshev_polynomial_u import ChebyshevPolynomialU


def chebyshev_polynomial_u_to_polynomial(
    c: ChebyshevPolynomialU,
) -> Polynomial:
    """Convert Chebyshev U series to power polynomial.

    Parameters
    ----------
    c : ChebyshevPolynomialU
        Chebyshev U series.

    Returns
    -------
    Polynomial
        Equivalent power polynomial.

    Notes
    -----
    Uses the recurrence relation:
        U_0(x) = 1
        U_1(x) = 2x
        U_{n+1}(x) = 2*x*U_n(x) - U_{n-1}(x)

    to build power representations of each U_k, then combines them.

    Examples
    --------
    >>> c = chebyshev_polynomial_u(torch.tensor([0.0, 0.0, 1.0]))  # U_2
    >>> p = chebyshev_polynomial_u_to_polynomial(c)
    >>> p  # U_2 = 4x^2 - 1
    Polynomial(tensor([-1.,  0.,  4.]))
    """
    coeffs = c.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Build power representations of U_k for k = 0, ..., n-1
    # U_k_power[j] is the coefficient of x^j in U_k
    U_power = []

    # U_0 = 1
    U_power.append(
        torch.tensor([1.0], dtype=coeffs.dtype, device=coeffs.device)
    )

    if n > 1:
        # U_1 = 2x
        U_power.append(
            torch.tensor([0.0, 2.0], dtype=coeffs.dtype, device=coeffs.device)
        )

    # Recurrence: U_{k+1} = 2*x*U_k - U_{k-1}
    for k in range(1, n - 1):
        U_k = U_power[k]
        U_km1 = U_power[k - 1]

        # 2*x*U_k: shift coefficients and multiply by 2
        U_k_shifted = torch.zeros(
            len(U_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        U_k_shifted[1:] = 2.0 * U_k

        # Pad U_{k-1} to same length
        U_km1_padded = torch.zeros(
            len(U_k_shifted), dtype=coeffs.dtype, device=coeffs.device
        )
        U_km1_padded[: len(U_km1)] = U_km1

        U_kp1 = U_k_shifted - U_km1_padded
        U_power.append(U_kp1)

    # Combine: p = sum(c_k * U_k)
    # Result has degree n-1
    result = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)

    for k in range(n):
        c_k = coeffs[..., k]
        U_k = U_power[k]
        result[: len(U_k)] = result[: len(U_k)] + c_k * U_k

    return polynomial(result)

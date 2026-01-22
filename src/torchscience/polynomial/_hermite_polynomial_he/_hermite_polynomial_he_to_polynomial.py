import torch

from torchscience.polynomial._polynomial._polynomial import (
    Polynomial,
    polynomial,
)

from ._hermite_polynomial_he import HermitePolynomialHe


def hermite_polynomial_he_to_polynomial(
    c: HermitePolynomialHe,
) -> Polynomial:
    """Convert Probabilists' Hermite series to power polynomial.

    Parameters
    ----------
    c : HermitePolynomialHe
        Hermite series.

    Returns
    -------
    Polynomial
        Equivalent power polynomial.

    Notes
    -----
    Uses the recurrence relation:
        He_0(x) = 1
        He_1(x) = x
        He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)

    to build power representations of each He_k, then combines them.

    Examples
    --------
    >>> c = hermite_polynomial_he(torch.tensor([0.0, 0.0, 1.0]))  # He_2
    >>> p = hermite_polynomial_he_to_polynomial(c)
    >>> p  # He_2 = x^2 - 1
    Polynomial(tensor([-1.,  0.,  1.]))
    """
    # Convert to plain tensor for operations
    coeffs = c.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Build power representations of He_k for k = 0, ..., n-1
    # He_k_power[j] is the coefficient of x^j in He_k
    He_power = []

    # He_0 = 1
    He_power.append(
        torch.tensor([1.0], dtype=coeffs.dtype, device=coeffs.device)
    )

    if n > 1:
        # He_1 = x
        He_power.append(
            torch.tensor([0.0, 1.0], dtype=coeffs.dtype, device=coeffs.device)
        )

    # Recurrence: He_{k+1} = x * He_k - k * He_{k-1}
    for k in range(1, n - 1):
        He_k = He_power[k]
        He_km1 = He_power[k - 1]

        # x * He_k: shift coefficients
        He_k_shifted = torch.zeros(
            len(He_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        He_k_shifted[1:] = He_k

        # Pad He_{k-1} to same length and multiply by k
        He_km1_padded = torch.zeros(
            len(He_k_shifted), dtype=coeffs.dtype, device=coeffs.device
        )
        He_km1_padded[: len(He_km1)] = k * He_km1

        He_kp1 = He_k_shifted - He_km1_padded
        He_power.append(He_kp1)

    # Combine: p = sum(c_k * He_k)
    # Result has degree n-1
    result = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)

    for k in range(n):
        c_k = coeffs[..., k]
        He_k = He_power[k]
        result[: len(He_k)] = result[: len(He_k)] + c_k * He_k

    return polynomial(result)

import torch

from torchscience.polynomial._polynomial import Polynomial

from ._laguerre_polynomial_l import LaguerrePolynomialL


def laguerre_polynomial_l_to_polynomial(
    c: LaguerrePolynomialL,
) -> Polynomial:
    """Convert Laguerre series to power polynomial.

    Parameters
    ----------
    c : LaguerrePolynomialL
        Laguerre series.

    Returns
    -------
    Polynomial
        Equivalent power polynomial.

    Notes
    -----
    Uses the recurrence relation:
        L_0(x) = 1
        L_1(x) = 1 - x
        (k+1) L_{k+1}(x) = (2k+1-x) L_k(x) - k L_{k-1}(x)

    to build power representations of each L_k, then combines them.

    Examples
    --------
    >>> c = laguerre_polynomial_l(torch.tensor([0.0, 0.0, 1.0]))  # L_2
    >>> p = laguerre_polynomial_l_to_polynomial(c)
    >>> p  # L_2 = 1 - 2x + x^2/2
    Polynomial(tensor([1.0000, -2.0000,  0.5000]))
    """
    coeffs = c.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Build power representations of L_k for k = 0, ..., n-1
    # L_k_power[j] is the coefficient of x^j in L_k
    L_power = []

    # L_0 = 1
    L_power.append(
        torch.tensor([1.0], dtype=coeffs.dtype, device=coeffs.device)
    )

    if n > 1:
        # L_1 = 1 - x
        L_power.append(
            torch.tensor([1.0, -1.0], dtype=coeffs.dtype, device=coeffs.device)
        )

    # Recurrence: (k+1) L_{k+1} = (2k+1-x) L_k - k L_{k-1}
    #             (k+1) L_{k+1} = (2k+1) L_k - x L_k - k L_{k-1}
    # L_{k+1} = [(2k+1) L_k - x L_k - k L_{k-1}] / (k+1)
    for k in range(1, n - 1):
        L_k = L_power[k]
        L_km1 = L_power[k - 1]

        # (2k+1) L_k
        term1 = (2 * k + 1) * L_k

        # x L_k: shift coefficients (multiply by x)
        L_k_shifted = torch.zeros(
            len(L_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        L_k_shifted[1:] = L_k

        # k L_{k-1}: pad to length len(L_k) + 1
        L_km1_padded = torch.zeros(
            len(L_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        L_km1_padded[: len(L_km1)] = k * L_km1

        # Pad term1 to same length
        term1_padded = torch.zeros(
            len(L_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        term1_padded[: len(term1)] = term1

        L_kp1 = (term1_padded - L_k_shifted - L_km1_padded) / (k + 1)
        L_power.append(L_kp1)

    # Combine: p = sum(c_k * L_k)
    # Result has degree n-1
    result = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)

    for k in range(n):
        c_k = coeffs[..., k]
        L_k = L_power[k]
        result[: len(L_k)] = result[: len(L_k)] + c_k * L_k

    return Polynomial(coeffs=result)

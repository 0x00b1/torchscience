import torch

from torchscience.polynomial._polynomial import Polynomial

from ._hermite_polynomial_h import HermitePolynomialH


def hermite_polynomial_h_to_polynomial(
    c: HermitePolynomialH,
) -> Polynomial:
    """Convert Physicists' Hermite series to power polynomial.

    Parameters
    ----------
    c : HermitePolynomialH
        Hermite series.

    Returns
    -------
    Polynomial
        Equivalent power polynomial.

    Notes
    -----
    Uses the recurrence relation:
        H_0(x) = 1
        H_1(x) = 2x
        H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)

    to build power representations of each H_k, then combines them.

    Examples
    --------
    >>> c = hermite_polynomial_h(torch.tensor([0.0, 0.0, 1.0]))  # H_2
    >>> p = hermite_polynomial_h_to_polynomial(c)
    >>> p.coeffs  # H_2 = 4x^2 - 2
    tensor([-2.,  0.,  4.])
    """
    coeffs = c.coeffs
    n = coeffs.shape[-1]

    # Build power representations of H_k for k = 0, ..., n-1
    # H_k_power[j] is the coefficient of x^j in H_k
    H_power = []

    # H_0 = 1
    H_power.append(
        torch.tensor([1.0], dtype=coeffs.dtype, device=coeffs.device)
    )

    if n > 1:
        # H_1 = 2x
        H_power.append(
            torch.tensor([0.0, 2.0], dtype=coeffs.dtype, device=coeffs.device)
        )

    # Recurrence: H_{k+1} = 2x * H_k - 2k * H_{k-1}
    for k in range(1, n - 1):
        H_k = H_power[k]
        H_km1 = H_power[k - 1]

        # 2x * H_k: shift coefficients and multiply by 2
        H_k_shifted = torch.zeros(
            len(H_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        H_k_shifted[1:] = 2.0 * H_k

        # Pad H_{k-1} to same length and multiply by 2k
        H_km1_padded = torch.zeros(
            len(H_k_shifted), dtype=coeffs.dtype, device=coeffs.device
        )
        H_km1_padded[: len(H_km1)] = 2.0 * k * H_km1

        H_kp1 = H_k_shifted - H_km1_padded
        H_power.append(H_kp1)

    # Combine: p = sum(c_k * H_k)
    # Result has degree n-1
    result = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)

    for k in range(n):
        c_k = coeffs[..., k]
        H_k = H_power[k]
        result[: len(H_k)] = result[: len(H_k)] + c_k * H_k

    return Polynomial(coeffs=result)

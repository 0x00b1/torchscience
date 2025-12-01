import torch

from torchscience.polynomial._polynomial import Polynomial

from ._chebyshev_polynomial_w import ChebyshevPolynomialW


def chebyshev_polynomial_w_to_polynomial(
    c: ChebyshevPolynomialW,
) -> Polynomial:
    """Convert Chebyshev W series to power polynomial.

    Parameters
    ----------
    c : ChebyshevPolynomialW
        Chebyshev W series.

    Returns
    -------
    Polynomial
        Equivalent power polynomial.

    Notes
    -----
    Uses the recurrence relation:
        W_0(x) = 1
        W_1(x) = 2x + 1
        W_{n+1}(x) = 2*x*W_n(x) - W_{n-1}(x)

    to build power representations of each W_k, then combines them.

    Examples
    --------
    >>> c = ChebyshevPolynomialW(coeffs=torch.tensor([0.0, 0.0, 1.0]))  # W_2
    >>> p = chebyshev_polynomial_w_to_polynomial(c)
    >>> p.coeffs  # W_2 = 4x^2 + 2x - 1
    tensor([-1.,  2.,  4.])
    """
    coeffs = c.coeffs
    n = coeffs.shape[-1]

    # Build power representations of W_k for k = 0, ..., n-1
    # W_k_power[j] is the coefficient of x^j in W_k
    W_power = []

    # W_0 = 1
    W_power.append(
        torch.tensor([1.0], dtype=coeffs.dtype, device=coeffs.device)
    )

    if n > 1:
        # W_1 = 2x + 1
        W_power.append(
            torch.tensor([1.0, 2.0], dtype=coeffs.dtype, device=coeffs.device)
        )

    # Recurrence: W_{k+1} = 2*x*W_k - W_{k-1}
    for k in range(1, n - 1):
        W_k = W_power[k]
        W_km1 = W_power[k - 1]

        # 2*x*W_k: shift coefficients and multiply by 2
        W_k_shifted = torch.zeros(
            len(W_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        W_k_shifted[1:] = 2.0 * W_k

        # Pad W_{k-1} to same length
        W_km1_padded = torch.zeros(
            len(W_k_shifted), dtype=coeffs.dtype, device=coeffs.device
        )
        W_km1_padded[: len(W_km1)] = W_km1

        W_kp1 = W_k_shifted - W_km1_padded
        W_power.append(W_kp1)

    # Combine: p = sum(c_k * W_k)
    # Result has degree n-1
    result = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)

    for k in range(n):
        c_k = coeffs[..., k]
        W_k = W_power[k]
        result[: len(W_k)] = result[: len(W_k)] + c_k * W_k

    return Polynomial(coeffs=result)

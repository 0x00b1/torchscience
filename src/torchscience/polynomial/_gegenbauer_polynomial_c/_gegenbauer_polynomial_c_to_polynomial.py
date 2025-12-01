import torch

from torchscience.polynomial._polynomial import Polynomial

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_to_polynomial(
    c: GegenbauerPolynomialC,
) -> Polynomial:
    """Convert Gegenbauer series to power polynomial.

    Parameters
    ----------
    c : GegenbauerPolynomialC
        Gegenbauer series.

    Returns
    -------
    Polynomial
        Equivalent power polynomial.

    Notes
    -----
    Uses the recurrence relation:
        C_0^{lambda}(x) = 1
        C_1^{lambda}(x) = 2*lambda*x
        C_{n+1}^{lambda}(x) = (2*(n+lambda)/(n+1))*x*C_n - ((n+2*lambda-1)/(n+1))*C_{n-1}

    to build power representations of each C_k^{lambda}, then combines them.

    Examples
    --------
    >>> c = gegenbauer_polynomial_c(
    ...     torch.tensor([0.0, 0.0, 1.0]), torch.tensor(1.0)
    ... )  # C_2^1
    >>> p = gegenbauer_polynomial_c_to_polynomial(c)
    >>> p.coeffs  # C_2^1 = 2*(2x^2 - 1) = 4x^2 - 2
    tensor([-1.,  0.,  4.])
    """
    coeffs = c.coeffs
    lambda_ = c.lambda_
    n = coeffs.shape[-1]

    # Get lambda as scalar
    lambda_val = lambda_.item() if lambda_.dim() == 0 else lambda_[0].item()

    # Build power representations of C_k^{lambda} for k = 0, ..., n-1
    # C_k_power[j] is the coefficient of x^j in C_k^{lambda}
    C_power = []

    # C_0^{lambda} = 1
    C_power.append(
        torch.tensor([1.0], dtype=coeffs.dtype, device=coeffs.device)
    )

    if n > 1:
        # C_1^{lambda} = 2*lambda*x
        C_power.append(
            torch.tensor(
                [0.0, 2.0 * lambda_val],
                dtype=coeffs.dtype,
                device=coeffs.device,
            )
        )

    # Recurrence: C_{k+1} = (2*(k+lambda)/(k+1))*x*C_k - ((k+2*lambda-1)/(k+1))*C_{k-1}
    for k in range(1, n - 1):
        C_k = C_power[k]
        C_km1 = C_power[k - 1]

        # Coefficients in recurrence
        a_k = 2.0 * (k + lambda_val) / (k + 1)
        b_k = (k + 2.0 * lambda_val - 1.0) / (k + 1)

        # a_k * x * C_k: shift coefficients and multiply
        C_k_shifted = torch.zeros(
            len(C_k) + 1, dtype=coeffs.dtype, device=coeffs.device
        )
        C_k_shifted[1:] = a_k * C_k

        # Pad C_{k-1} to same length
        C_km1_padded = torch.zeros(
            len(C_k_shifted), dtype=coeffs.dtype, device=coeffs.device
        )
        C_km1_padded[: len(C_km1)] = b_k * C_km1

        C_kp1 = C_k_shifted - C_km1_padded
        C_power.append(C_kp1)

    # Combine: p = sum(c_k * C_k^{lambda})
    # Result has degree n-1
    result = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)

    for k in range(n):
        c_k = coeffs[..., k]
        C_k = C_power[k]
        result[: len(C_k)] = result[: len(C_k)] + c_k * C_k

    return Polynomial(coeffs=result)

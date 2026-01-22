import torch

from ._legendre_polynomial_p import (
    LegendrePolynomialP,
    legendre_polynomial_p,
)


def legendre_polynomial_p_mulx(
    a: LegendrePolynomialP,
) -> LegendrePolynomialP:
    """Multiply Legendre series by x.

    Uses the recurrence relation:
        x * P_k(x) = [(k+1)*P_{k+1}(x) + k*P_{k-1}(x)] / (2k+1)

    Parameters
    ----------
    a : LegendrePolynomialP
        Series to multiply by x.

    Returns
    -------
    LegendrePolynomialP
        Series representing x * a(x).

    Notes
    -----
    The degree increases by 1.

    The Legendre recurrence relation is:
        (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)

    Solving for x*P_n(x):
        x*P_n(x) = [(n+1)*P_{n+1}(x) + n*P_{n-1}(x)] / (2n+1)

    Examples
    --------
    >>> a = legendre_polynomial_p(torch.tensor([1.0]))  # P_0
    >>> b = legendre_polynomial_p_mulx(a)
    >>> b  # x * P_0 = P_1
    LegendrePolynomialP(tensor([0., 1.]))
    """
    coeffs = a.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Result has one more coefficient
    result_shape = list(coeffs.shape)
    result_shape[-1] = n + 1
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # x * P_0 = P_1 contributes c_0 to coefficient 1
    result[..., 1] = result[..., 1] + coeffs[..., 0]

    # x * P_k = [(k+1)*P_{k+1} + k*P_{k-1}] / (2k+1) for k >= 1
    for k in range(1, n):
        c_k = coeffs[..., k]
        denom = 2 * k + 1

        # Contribution to P_{k-1} coefficient
        result[..., k - 1] = result[..., k - 1] + (k / denom) * c_k
        # Contribution to P_{k+1} coefficient
        result[..., k + 1] = result[..., k + 1] + ((k + 1) / denom) * c_k

    return legendre_polynomial_p(result)

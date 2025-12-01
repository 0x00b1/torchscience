import torch

from ._chebyshev_polynomial_v import ChebyshevPolynomialV


def chebyshev_polynomial_v_mulx(
    a: ChebyshevPolynomialV,
) -> ChebyshevPolynomialV:
    """Multiply Chebyshev V series by x.

    Uses the recurrence relation:
        x * V_k(x) = 0.5 * (V_{k-1}(x) + V_{k+1}(x)) + 0.5 * V_k(x)  for k >= 1
        x * V_0(x) = 0.5 * (V_0(x) + V_1(x))

    Parameters
    ----------
    a : ChebyshevPolynomialV
        Series to multiply by x.

    Returns
    -------
    ChebyshevPolynomialV
        Series representing x * a(x).

    Notes
    -----
    The degree increases by 1.

    For Chebyshev V polynomials, the multiplication by x relation is:
        x * V_n(x) = 0.5 * (V_{n+1}(x) + V_{n-1}(x)) + 0.5 * V_n(x)

    which can be derived from the recurrence V_{n+1} = 2x*V_n - V_{n-1}.

    Examples
    --------
    >>> a = chebyshev_polynomial_v(torch.tensor([1.0]))  # V_0
    >>> b = chebyshev_polynomial_v_mulx(a)
    >>> b.coeffs  # x * V_0 = 0.5 * (V_0 + V_1)
    tensor([0.5, 0.5])
    """
    coeffs = a.coeffs
    n = coeffs.shape[-1]

    # Result has one more coefficient
    result_shape = list(coeffs.shape)
    result_shape[-1] = n + 1
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # For V polynomials: x * V_k = 0.5 * (V_{k-1} + V_{k+1}) + 0.5 * V_k  for k >= 1
    # x * V_0 = 0.5 * (V_0 + V_1)

    # Handle V_0 term: x * V_0 = 0.5 * V_0 + 0.5 * V_1
    result[..., 0] = result[..., 0] + 0.5 * coeffs[..., 0]
    result[..., 1] = result[..., 1] + 0.5 * coeffs[..., 0]

    # Handle V_k terms for k >= 1
    for k in range(1, n):
        c_k = coeffs[..., k]
        # x * V_k contributes to V_{k-1}, V_k, and V_{k+1}
        result[..., k - 1] = result[..., k - 1] + 0.5 * c_k
        result[..., k] = result[..., k] + 0.5 * c_k
        result[..., k + 1] = result[..., k + 1] + 0.5 * c_k

    return ChebyshevPolynomialV(coeffs=result)

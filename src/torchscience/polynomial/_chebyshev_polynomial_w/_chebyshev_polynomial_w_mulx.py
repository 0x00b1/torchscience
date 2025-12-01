import torch

from ._chebyshev_polynomial_w import ChebyshevPolynomialW


def chebyshev_polynomial_w_mulx(
    a: ChebyshevPolynomialW,
) -> ChebyshevPolynomialW:
    """Multiply Chebyshev W series by x.

    Uses the recurrence relation:
        x * W_k(x) = 0.5 * (W_{k-1}(x) + W_{k+1}(x))  for k >= 1
        x * W_0(x) = 0.5 * (W_1(x) - W_0(x))

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Series to multiply by x.

    Returns
    -------
    ChebyshevPolynomialW
        Series representing x * a(x).

    Notes
    -----
    The degree increases by 1.

    For Chebyshev W polynomials, the multiplication by x relation is derived
    from the recurrence W_{n+1} = 2x*W_n - W_{n-1} and W_1 = 2x + 1.

    Since W_0 = 1 and W_1 = 2x + 1, we have x = 0.5*(W_1 - 1) = 0.5*(W_1 - W_0).

    Examples
    --------
    >>> a = ChebyshevPolynomialW(coeffs=torch.tensor([1.0]))  # W_0
    >>> b = chebyshev_polynomial_w_mulx(a)
    >>> b.coeffs  # x * W_0 = 0.5 * (W_1 - W_0)
    tensor([-0.5, 0.5])
    """
    coeffs = a.coeffs
    n = coeffs.shape[-1]

    # Result has one more coefficient
    result_shape = list(coeffs.shape)
    result_shape[-1] = n + 1
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # For W polynomials:
    # x * W_0 = 0.5 * (W_1 - W_0)
    # x * W_k = 0.5 * (W_{k-1} + W_{k+1})  for k >= 1

    # Handle W_0 term: x * W_0 = -0.5 * W_0 + 0.5 * W_1
    result[..., 0] = result[..., 0] - 0.5 * coeffs[..., 0]
    result[..., 1] = result[..., 1] + 0.5 * coeffs[..., 0]

    # Handle W_k terms for k >= 1
    for k in range(1, n):
        c_k = coeffs[..., k]
        # x * W_k contributes to W_{k-1} and W_{k+1}
        result[..., k - 1] = result[..., k - 1] + 0.5 * c_k
        result[..., k + 1] = result[..., k + 1] + 0.5 * c_k

    return ChebyshevPolynomialW(coeffs=result)

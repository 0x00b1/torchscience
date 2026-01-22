import torch

from ._chebyshev_polynomial_t import (
    ChebyshevPolynomialT,
    chebyshev_polynomial_t,
)


def chebyshev_polynomial_t_mulx(
    a: ChebyshevPolynomialT,
) -> ChebyshevPolynomialT:
    """Multiply Chebyshev series by x.

    Uses the recurrence relation:
        x * T_k(x) = 0.5 * (T_{k-1}(x) + T_{k+1}(x))  for k >= 1
        x * T_0(x) = T_1(x)

    Parameters
    ----------
    a : ChebyshevPolynomialT
        Series to multiply by x.

    Returns
    -------
    ChebyshevPolynomialT
        Series representing x * a(x).

    Notes
    -----
    The degree increases by 1.

    Examples
    --------
    >>> a = chebyshev_polynomial_t(torch.tensor([1.0]))  # T_0
    >>> b = chebyshev_polynomial_t_mulx(a)
    >>> b  # x * T_0 = T_1
    ChebyshevPolynomialT(tensor([0., 1.]))
    """
    # The polynomial IS the coefficients tensor
    coeffs = a.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Result has one more coefficient
    result_shape = list(coeffs.shape)
    result_shape[-1] = n + 1
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # x * T_0 = T_1 contributes c_0 to coefficient 1
    result[..., 1] = result[..., 1] + coeffs[..., 0]

    # x * T_k = 0.5*(T_{k-1} + T_{k+1}) for k >= 1
    for k in range(1, n):
        c_k = coeffs[..., k]
        result[..., k - 1] = result[..., k - 1] + 0.5 * c_k
        result[..., k + 1] = result[..., k + 1] + 0.5 * c_k

    return chebyshev_polynomial_t(result)

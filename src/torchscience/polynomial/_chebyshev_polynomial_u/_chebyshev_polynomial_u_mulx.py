import torch

from ._chebyshev_polynomial_u import (
    ChebyshevPolynomialU,
    chebyshev_polynomial_u,
)


def chebyshev_polynomial_u_mulx(
    a: ChebyshevPolynomialU,
) -> ChebyshevPolynomialU:
    """Multiply Chebyshev U series by x.

    Uses the recurrence relation:
        x * U_k(x) = 0.5 * (U_{k-1}(x) + U_{k+1}(x))  for k >= 1
        x * U_0(x) = 0.5 * U_1(x)  (since U_1 = 2x, so x = U_1/2)

    Parameters
    ----------
    a : ChebyshevPolynomialU
        Series to multiply by x.

    Returns
    -------
    ChebyshevPolynomialU
        Series representing x * a(x).

    Notes
    -----
    The degree increases by 1.

    Examples
    --------
    >>> a = chebyshev_polynomial_u(torch.tensor([1.0]))  # U_0 = 1
    >>> b = chebyshev_polynomial_u_mulx(a)
    >>> b  # x * U_0 = x = U_1/2
    ChebyshevPolynomialU(tensor([0.0, 0.5]))
    """
    coeffs = a.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Result has one more coefficient
    result_shape = list(coeffs.shape)
    result_shape[-1] = n + 1
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # x * U_k = 0.5*(U_{k-1} + U_{k+1})
    # Special case: x * U_0 = x = U_1/2, so contributes 0.5 to U_1
    # For k >= 1: x * U_k contributes 0.5 to both U_{k-1} and U_{k+1}

    for k in range(n):
        c_k = coeffs[..., k]
        if k == 0:
            # x * U_0 = U_1/2 (since x = U_1/2)
            result[..., 1] = result[..., 1] + 0.5 * c_k
        else:
            # x * U_k = 0.5*(U_{k-1} + U_{k+1})
            result[..., k - 1] = result[..., k - 1] + 0.5 * c_k
            result[..., k + 1] = result[..., k + 1] + 0.5 * c_k

    return chebyshev_polynomial_u(result)

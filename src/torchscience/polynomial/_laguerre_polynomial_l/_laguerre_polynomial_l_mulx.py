import torch

from ._laguerre_polynomial_l import (
    LaguerrePolynomialL,
    laguerre_polynomial_l,
)


def laguerre_polynomial_l_mulx(
    a: LaguerrePolynomialL,
) -> LaguerrePolynomialL:
    """Multiply Laguerre series by x.

    Uses the recurrence relation:
        x * L_k(x) = (2k+1) L_k(x) - (k+1) L_{k+1}(x) - k L_{k-1}(x)

    Parameters
    ----------
    a : LaguerrePolynomialL
        Series to multiply by x.

    Returns
    -------
    LaguerrePolynomialL
        Series representing x * a(x).

    Notes
    -----
    The degree increases by 1.

    The Laguerre recurrence relation is:
        (k+1) L_{k+1}(x) = (2k+1-x) L_k(x) - k L_{k-1}(x)

    Solving for x*L_k(x):
        x L_k(x) = (2k+1) L_k(x) - (k+1) L_{k+1}(x) - k L_{k-1}(x)

    Examples
    --------
    >>> a = laguerre_polynomial_l(torch.tensor([1.0]))  # L_0
    >>> b = laguerre_polynomial_l_mulx(a)
    >>> b  # x * L_0 = x = L_0 - L_1
    LaguerrePolynomialL(tensor([ 1., -1.]))
    """
    coeffs = a.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Result has one more coefficient
    result_shape = list(coeffs.shape)
    result_shape[-1] = n + 1
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # x * L_k = (2k+1) L_k - (k+1) L_{k+1} - k L_{k-1}
    for k in range(n):
        c_k = coeffs[..., k]

        # Contribution to L_k coefficient: (2k+1) * c_k
        result[..., k] = result[..., k] + (2 * k + 1) * c_k

        # Contribution to L_{k+1} coefficient: -(k+1) * c_k
        result[..., k + 1] = result[..., k + 1] - (k + 1) * c_k

        # Contribution to L_{k-1} coefficient: -k * c_k (only if k >= 1)
        if k >= 1:
            result[..., k - 1] = result[..., k - 1] - k * c_k

    return laguerre_polynomial_l(result)

import torch

from ._laguerre_polynomial_l import LaguerrePolynomialL
from ._laguerre_polynomial_l_evaluate import laguerre_polynomial_l_evaluate


def laguerre_polynomial_l_antiderivative(
    a: LaguerrePolynomialL,
    order: int = 1,
    constant: float = 0.0,
) -> LaguerrePolynomialL:
    """Compute antiderivative of Laguerre series.

    Uses the formula for integrating Laguerre polynomials:
        integral(L_n(x)) = L_n(x) - L_{n+1}(x)

    The constant of integration is chosen such that the antiderivative
    evaluates to `constant` at x=0, matching NumPy's lagint behavior.

    Parameters
    ----------
    a : LaguerrePolynomialL
        Series to integrate.
    order : int, optional
        Order of integration. Default is 1.
    constant : float, optional
        Integration constant. The antiderivative will evaluate to this
        value at x=0. Default is 0.0.

    Returns
    -------
    LaguerrePolynomialL
        Antiderivative series.

    Notes
    -----
    The degree increases by 1 for each integration.

    For a Laguerre series f(x) = sum c_k L_k(x), the antiderivative is
    computed using:
        integral(L_n(x)) = L_n(x) - L_{n+1}(x)

    So F(x) = sum c_k * (L_k - L_{k+1}) = sum (c_k - c_{k-1}) L_k - c_{n-1} L_n
    where c_{-1} = 0.

    Examples
    --------
    >>> a = laguerre_polynomial_l(torch.tensor([1.0]))  # L_0 = 1
    >>> ia = laguerre_polynomial_l_antiderivative(a)
    >>> ia.coeffs  # integral(L_0) = L_0 - L_1
    tensor([ 1., -1.])
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return LaguerrePolynomialL(coeffs=a.coeffs.clone())

    coeffs = a.coeffs

    # Apply antiderivative 'order' times
    for i in range(order):
        n = coeffs.shape[-1]

        # Result has n+1 coefficients (degree increases by 1)
        result_shape = list(coeffs.shape)
        result_shape[-1] = n + 1
        i_coeffs = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # For Laguerre polynomials:
        # integral(L_k) dx = L_k - L_{k+1}
        #
        # So if we have f(x) = sum_{k=0}^{n-1} c_k L_k(x)
        # F(x) = integral f(x) dx = sum_{k=0}^{n-1} c_k * (L_k - L_{k+1})
        #      = sum_{k=0}^{n-1} c_k L_k - sum_{k=0}^{n-1} c_k L_{k+1}
        #      = sum_{k=0}^{n-1} c_k L_k - sum_{j=1}^{n} c_{j-1} L_j
        #
        # Coefficient of L_0: c_0
        # Coefficient of L_k (1 <= k <= n-1): c_k - c_{k-1}
        # Coefficient of L_n: -c_{n-1}

        # k = 0
        i_coeffs[..., 0] = coeffs[..., 0]

        # k = 1, ..., n-1
        for k in range(1, n):
            i_coeffs[..., k] = coeffs[..., k] - coeffs[..., k - 1]

        # k = n
        i_coeffs[..., n] = -coeffs[..., n - 1]

        # Set constant of integration so F(0) = constant (for first integration)
        # or F(0) = 0 (for subsequent integrations)
        k_val = constant if i == 0 else 0.0
        temp = LaguerrePolynomialL(coeffs=i_coeffs)
        x_zero = torch.zeros((), dtype=coeffs.dtype, device=coeffs.device)
        val_at_zero = laguerre_polynomial_l_evaluate(temp, x_zero)
        # L_k(0) = 1 for all k, so adding delta to i_coeffs[..., 0] shifts F(0) by delta
        i_coeffs[..., 0] = i_coeffs[..., 0] + (k_val - val_at_zero)

        coeffs = i_coeffs

    return LaguerrePolynomialL(coeffs=coeffs)

import torch

from ._legendre_polynomial_p import LegendrePolynomialP
from ._legendre_polynomial_p_evaluate import legendre_polynomial_p_evaluate


def legendre_polynomial_p_antiderivative(
    a: LegendrePolynomialP,
    order: int = 1,
    constant: float = 0.0,
) -> LegendrePolynomialP:
    """Compute antiderivative of Legendre series.

    Uses the formula for integrating Legendre polynomials:
        integral(P_n(x)) = (P_{n+1}(x) - P_{n-1}(x)) / (2n+1)

    The constant of integration is chosen such that the antiderivative
    evaluates to `constant` at x=0, matching NumPy's legint behavior.

    Parameters
    ----------
    a : LegendrePolynomialP
        Series to integrate.
    order : int, optional
        Order of integration. Default is 1.
    constant : float, optional
        Integration constant. The antiderivative will evaluate to this
        value at x=0. Default is 0.0.

    Returns
    -------
    LegendrePolynomialP
        Antiderivative series.

    Notes
    -----
    The degree increases by 1 for each integration.

    For a Legendre series f(x) = sum c_k P_k(x), the antiderivative is
    computed using:
        integral(P_n(x)) = (P_{n+1}(x) - P_{n-1}(x)) / (2n+1)

    Special case: integral(P_0(x)) = integral(1) = x = P_1(x)

    Examples
    --------
    >>> a = legendre_polynomial_p(torch.tensor([1.0]))  # P_0 = 1
    >>> ia = legendre_polynomial_p_antiderivative(a)
    >>> ia.coeffs  # integral(1) = x = P_1
    tensor([0., 1.])
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return LegendrePolynomialP(coeffs=a.coeffs.clone())

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

        # For Legendre polynomials:
        # integral(P_k) dx = (P_{k+1} - P_{k-1}) / (2k+1)  for k >= 1
        # integral(P_0) dx = P_1 = x
        #
        # So if we have f(x) = sum_{k=0}^{n-1} c_k P_k(x)
        # F(x) = integral f(x) dx = sum_{k=0}^{n-1} c_k * integral(P_k) dx
        #
        # For k=0: c_0 * integral(P_0) = c_0 * P_1
        #          Contributes c_0 to coefficient of P_1
        #
        # For k>=1: c_k * (P_{k+1} - P_{k-1}) / (2k+1)
        #           Contributes c_k/(2k+1) to coefficient of P_{k+1}
        #           Contributes -c_k/(2k+1) to coefficient of P_{k-1}

        # k=0 term: integral(P_0) = P_1
        i_coeffs[..., 1] = i_coeffs[..., 1] + coeffs[..., 0]

        # k>=1 terms
        for k in range(1, n):
            factor = 1.0 / (2.0 * k + 1.0)
            # Contribution to P_{k+1}
            i_coeffs[..., k + 1] = (
                i_coeffs[..., k + 1] + coeffs[..., k] * factor
            )
            # Contribution to P_{k-1}
            i_coeffs[..., k - 1] = (
                i_coeffs[..., k - 1] - coeffs[..., k] * factor
            )

        # Set constant of integration so F(0) = constant (for first integration)
        # or F(0) = 0 (for subsequent integrations)
        k_val = constant if i == 0 else 0.0
        temp = LegendrePolynomialP(coeffs=i_coeffs)
        x_zero = torch.zeros((), dtype=coeffs.dtype, device=coeffs.device)
        val_at_zero = legendre_polynomial_p_evaluate(temp, x_zero)
        # P_0(x) = 1, so adding delta to i_coeffs[..., 0] shifts F(0) by delta
        i_coeffs[..., 0] = i_coeffs[..., 0] + (k_val - val_at_zero)

        coeffs = i_coeffs

    return LegendrePolynomialP(coeffs=coeffs)

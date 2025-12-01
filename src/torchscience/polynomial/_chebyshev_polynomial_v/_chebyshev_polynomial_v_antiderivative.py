import torch

from ._chebyshev_polynomial_v import ChebyshevPolynomialV
from ._chebyshev_polynomial_v_evaluate import chebyshev_polynomial_v_evaluate


def chebyshev_polynomial_v_antiderivative(
    a: ChebyshevPolynomialV,
    order: int = 1,
    constant: float = 0.0,
) -> ChebyshevPolynomialV:
    """Compute antiderivative of Chebyshev V series.

    The constant of integration is chosen such that the antiderivative
    evaluates to `constant` at x=0.

    Parameters
    ----------
    a : ChebyshevPolynomialV
        Series to integrate.
    order : int, optional
        Order of integration. Default is 1.
    constant : float, optional
        Integration constant. The antiderivative will evaluate to this
        value at x=0. Default is 0.0.

    Returns
    -------
    ChebyshevPolynomialV
        Antiderivative series.

    Notes
    -----
    The degree increases by 1 for each integration.

    For Chebyshev V polynomials, the antiderivative formula is:
        integral(V_n) = (V_{n+1} - V_{n-1}) / (2*(n+1)) for n >= 1
        integral(V_0) = (V_1 + V_0) / 2

    Examples
    --------
    >>> a = chebyshev_polynomial_v(torch.tensor([1.0]))  # constant 1 = V_0
    >>> ia = chebyshev_polynomial_v_antiderivative(a)
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return ChebyshevPolynomialV(coeffs=a.coeffs.clone())

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

        # For V polynomials, the integration formula follows from the derivative:
        # If d/dx V_n = 2*(V_{n-1} + V_{n-3} + ...), then
        # integral(V_n) involves V_{n+1} and lower order terms

        # Using the standard approach similar to Chebyshev T:
        # a_1 = c_0 - c_2/2 (adjusted for V polynomials)
        # a_k = (c_{k-1} - c_{k+1}) / (2k) for k >= 2

        # Handle c_0 term: integral(V_0) = (V_1 + V_0)/2
        # So c_0 contributes to both a_0 and a_1
        i_coeffs[..., 1] = coeffs[..., 0]
        if n > 2:
            i_coeffs[..., 1] = i_coeffs[..., 1] - coeffs[..., 2] / 2.0

        # a_k = (c_{k-1} - c_{k+1}) / (2k) for k >= 2
        for k in range(2, n + 1):
            c_km1 = coeffs[..., k - 1] if k - 1 < n else 0.0
            c_kp1 = coeffs[..., k + 1] if k + 1 < n else 0.0
            i_coeffs[..., k] = (c_km1 - c_kp1) / (2.0 * k)

        # Set a_0 so that the antiderivative evaluates to constant at x=0
        # (only for first integration; subsequent integrations use 0)
        k_val = constant if i == 0 else 0.0
        temp = ChebyshevPolynomialV(coeffs=i_coeffs)
        x_zero = torch.zeros((), dtype=coeffs.dtype, device=coeffs.device)
        val_at_zero = chebyshev_polynomial_v_evaluate(temp, x_zero)
        i_coeffs[..., 0] = k_val - val_at_zero

        coeffs = i_coeffs

    return ChebyshevPolynomialV(coeffs=coeffs)

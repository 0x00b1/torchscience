import torch

from ._chebyshev_polynomial_t import (
    ChebyshevPolynomialT,
    chebyshev_polynomial_t,
)
from ._chebyshev_polynomial_t_evaluate import chebyshev_polynomial_t_evaluate


def chebyshev_polynomial_t_antiderivative(
    a: ChebyshevPolynomialT,
    order: int = 1,
    constant: float = 0.0,
) -> ChebyshevPolynomialT:
    """Compute antiderivative of Chebyshev series.

    Uses the formula:
        a_1 = c_0 - c_2/2
        a_k = (c_{k-1} - c_{k+1}) / (2k)  for k >= 2
        a_0 = constant - p(0)  (where p(0) is the antiderivative at x=0)

    The constant of integration is chosen such that the antiderivative
    evaluates to `constant` at x=0, matching NumPy's chebint behavior.

    Parameters
    ----------
    a : ChebyshevPolynomialT
        Series to integrate.
    order : int, optional
        Order of integration. Default is 1.
    constant : float, optional
        Integration constant. The antiderivative will evaluate to this
        value at x=0. Default is 0.0.

    Returns
    -------
    ChebyshevPolynomialT
        Antiderivative series.

    Notes
    -----
    The degree increases by 1 for each integration.

    Examples
    --------
    >>> a = chebyshev_polynomial_t(torch.tensor([1.0]))  # constant 1 = T_0
    >>> ia = chebyshev_polynomial_t_antiderivative(a)
    >>> ia  # integral(1) = x = T_1
    ChebyshevPolynomialT(tensor([0., 1.]))
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return chebyshev_polynomial_t(a.clone())

    coeffs = a.as_subclass(torch.Tensor)

    # Apply antiderivative 'order' times
    for i in range(order):
        n = coeffs.shape[-1]

        # Result has n+1 coefficients (degree increases by 1)
        result_shape = list(coeffs.shape)
        result_shape[-1] = n + 1
        i_coeffs = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # a_1 = c_0 - c_2/2
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
        temp = chebyshev_polynomial_t(i_coeffs)
        x_zero = torch.zeros((), dtype=coeffs.dtype, device=coeffs.device)
        val_at_zero = chebyshev_polynomial_t_evaluate(temp, x_zero)
        i_coeffs[..., 0] = k_val - val_at_zero

        coeffs = i_coeffs

    return chebyshev_polynomial_t(coeffs)

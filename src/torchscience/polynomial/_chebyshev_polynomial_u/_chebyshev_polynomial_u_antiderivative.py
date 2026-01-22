import torch

from ._chebyshev_polynomial_u import (
    ChebyshevPolynomialU,
    chebyshev_polynomial_u,
)
from ._chebyshev_polynomial_u_evaluate import chebyshev_polynomial_u_evaluate


def chebyshev_polynomial_u_antiderivative(
    a: ChebyshevPolynomialU,
    order: int = 1,
    constant: float = 0.0,
) -> ChebyshevPolynomialU:
    """Compute antiderivative of Chebyshev U series.

    The antiderivative of U_n is computed using the relation:
        integral U_n dx = U_{n+1}/(2(n+1)) - U_{n-1}/(2(n-1)) for n >= 2
        integral U_0 dx = U_1/2
        integral U_1 dx = U_2/4

    The constant of integration is chosen such that the antiderivative
    evaluates to `constant` at x=0, matching NumPy's chebint behavior.

    Parameters
    ----------
    a : ChebyshevPolynomialU
        Series to integrate.
    order : int, optional
        Order of integration. Default is 1.
    constant : float, optional
        Integration constant. The antiderivative will evaluate to this
        value at x=0. Default is 0.0.

    Returns
    -------
    ChebyshevPolynomialU
        Antiderivative series.

    Notes
    -----
    The degree increases by 1 for each integration.

    Examples
    --------
    >>> a = chebyshev_polynomial_u(torch.tensor([1.0]))  # U_0 = 1
    >>> ia = chebyshev_polynomial_u_antiderivative(a)
    >>> ia  # integral(1) = x = U_1/2, so need 2*U_1
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return chebyshev_polynomial_u(a.clone())

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

        # For U polynomials:
        # integral U_0 dx = x = U_1/2 (since U_1 = 2x)
        # integral U_1 dx = x^2 = (U_0 + U_2)/4
        # integral U_n dx = U_{n+1}/(2(n+1)) - U_{n-1}/(2(n-1)) for n >= 2
        #
        # So coefficient k in antiderivative gets contributions from
        # coefficients k-1 and k+1 in original (with proper signs/factors)

        for k in range(n):
            c_k = coeffs[..., k]
            if k == 0:
                # integral c_0 * U_0 = c_0 * U_1 / 2
                i_coeffs[..., 1] = i_coeffs[..., 1] + c_k / 2.0
            elif k == 1:
                # integral c_1 * U_1 = c_1 * (U_0 + U_2) / 4
                i_coeffs[..., 0] = i_coeffs[..., 0] + c_k / 4.0
                i_coeffs[..., 2] = i_coeffs[..., 2] + c_k / 4.0
            else:
                # integral c_k * U_k = c_k * (U_{k+1}/(2(k+1)) - U_{k-1}/(2(k-1)))
                i_coeffs[..., k + 1] = i_coeffs[..., k + 1] + c_k / (
                    2.0 * (k + 1)
                )
                i_coeffs[..., k - 1] = i_coeffs[..., k - 1] - c_k / (
                    2.0 * (k - 1)
                )

        # Set constant term so that antiderivative evaluates to constant at x=0
        # (only for first integration; subsequent integrations use 0)
        k_val = constant if i == 0 else 0.0
        temp = chebyshev_polynomial_u(i_coeffs)
        x_zero = torch.zeros((), dtype=coeffs.dtype, device=coeffs.device)
        val_at_zero = chebyshev_polynomial_u_evaluate(temp, x_zero)
        i_coeffs[..., 0] = i_coeffs[..., 0] + (k_val - val_at_zero)

        coeffs = i_coeffs

    return chebyshev_polynomial_u(coeffs)

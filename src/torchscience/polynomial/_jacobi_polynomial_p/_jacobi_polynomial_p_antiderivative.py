import torch

from ._jacobi_polynomial_p import JacobiPolynomialP, jacobi_polynomial_p
from ._jacobi_polynomial_p_evaluate import jacobi_polynomial_p_evaluate


def jacobi_polynomial_p_antiderivative(
    a: JacobiPolynomialP,
    order: int = 1,
    constant: float = 0.0,
) -> JacobiPolynomialP:
    """Compute antiderivative of Jacobi series.

    The constant of integration is chosen such that the antiderivative
    evaluates to `constant` at x=0.

    Parameters
    ----------
    a : JacobiPolynomialP
        Series to integrate.
    order : int, optional
        Order of integration. Default is 1.
    constant : float, optional
        Integration constant. The antiderivative will evaluate to this
        value at x=0. Default is 0.0.

    Returns
    -------
    JacobiPolynomialP
        Antiderivative series.

    Notes
    -----
    The degree increases by 1 for each integration.

    For Jacobi polynomials with alpha=beta=0 (Legendre), the antiderivative formula is:
        integral P_n(x) dx = (P_{n+1}(x) - P_{n-1}(x)) / (2n+1)  for n >= 1
        integral P_0(x) dx = P_1(x)

    For general Jacobi polynomials, we use a similar recurrence-based approach.

    Examples
    --------
    >>> a = jacobi_polynomial_p(torch.tensor([1.0]), alpha=0.0, beta=0.0)  # P_0 = 1
    >>> ia = jacobi_polynomial_p_antiderivative(a)
    >>> ia  # integral(1) = x = P_1
    JacobiPolynomialP(tensor([0., 1.]), alpha=tensor(0.), beta=tensor(0.))
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    # Get coefficients as plain tensor
    coeffs = a.as_subclass(torch.Tensor)
    alpha = a.alpha
    beta = a.beta
    ab = alpha + beta

    if order == 0:
        return jacobi_polynomial_p(coeffs.clone(), alpha.clone(), beta.clone())

    # Apply antiderivative 'order' times
    for i in range(order):
        n = coeffs.shape[-1]

        # Result has n+1 coefficients (degree increases by 1)
        result_shape = list(coeffs.shape)
        result_shape[-1] = n + 1
        i_coeffs = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # For Jacobi polynomials, we use the formula:
        # integral P_k^{(alpha,beta)}(x) dx = (1/(2k+alpha+beta)) * [A_k * P_{k+1} - B_k * P_{k-1}]
        #
        # For k=0: integral P_0 dx = integral 1 dx = x
        # We express x in terms of Jacobi polynomials:
        # x = (beta-alpha + (alpha+beta+2)*P_1) / (alpha+beta+2)  rearranged from P_1 formula
        # So integral P_0 dx has coefficient 2/(alpha+beta+2) for P_1
        #
        # For k >= 1, using the integration formula and normalizing:
        # Contribution to P_{k+1}: 2/(2k+alpha+beta+2)
        # Contribution to P_{k-1}: -2(k+alpha)(k+beta)/[(2k+alpha+beta)(2k+alpha+beta+2)(k)]  [approx]

        # k=0 term: integral P_0 dx involves P_1
        # P_1^{(alpha,beta)}(x) = (alpha-beta)/2 + (alpha+beta+2)/2 * x
        # So x = 2*(P_1 - (alpha-beta)/2) / (alpha+beta+2) = 2*P_1/(alpha+beta+2) - (alpha-beta)/(alpha+beta+2)
        # But since we need integral 1 dx = x expressed in Jacobi basis:
        # We need the coefficient for P_1 to be 2/(alpha+beta+2)
        i_coeffs[..., 1] = i_coeffs[..., 1] + coeffs[..., 0] * 2.0 / (ab + 2)

        # k>=1 terms: use generalization of Legendre formula
        for k in range(1, n):
            two_k_ab = 2.0 * k + ab
            # Main contribution to P_{k+1}
            factor_up = 2.0 / (two_k_ab + 2)
            i_coeffs[..., k + 1] = (
                i_coeffs[..., k + 1] + coeffs[..., k] * factor_up
            )

            # Contribution to P_{k-1} (analogous to Legendre case)
            factor_down = -2.0 / two_k_ab
            i_coeffs[..., k - 1] = (
                i_coeffs[..., k - 1] + coeffs[..., k] * factor_down
            )

        # Set constant of integration so F(0) = constant (for first integration)
        # or F(0) = 0 (for subsequent integrations)
        k_val = constant if i == 0 else 0.0
        temp = jacobi_polynomial_p(i_coeffs, alpha.clone(), beta.clone())
        x_zero = torch.zeros((), dtype=coeffs.dtype, device=coeffs.device)
        val_at_zero = jacobi_polynomial_p_evaluate(temp, x_zero)
        # P_0(x) = 1, so adding delta to i_coeffs[..., 0] shifts F(0) by delta
        i_coeffs[..., 0] = i_coeffs[..., 0] + (k_val - val_at_zero)

        coeffs = i_coeffs

    return jacobi_polynomial_p(coeffs, alpha.clone(), beta.clone())

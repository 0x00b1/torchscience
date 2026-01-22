import torch

from ._jacobi_polynomial_p import JacobiPolynomialP, jacobi_polynomial_p


def jacobi_polynomial_p_mulx(
    a: JacobiPolynomialP,
) -> JacobiPolynomialP:
    """Multiply Jacobi series by x.

    Uses the recurrence relation for Jacobi polynomials to express
    x * P_k^{(alpha,beta)}(x) as a linear combination of P_{k-1}, P_k, P_{k+1}.

    Parameters
    ----------
    a : JacobiPolynomialP
        Series to multiply by x.

    Returns
    -------
    JacobiPolynomialP
        Series representing x * a(x).

    Notes
    -----
    The degree increases by 1.

    The Jacobi recurrence relation is:
        P_{n+1}^{(alpha,beta)}(x) = (A_n + B_n * x) * P_n^{(alpha,beta)}(x) - C_n * P_{n-1}^{(alpha,beta)}(x)

    where:
        A_n = (alpha^2 - beta^2) / ((2n + alpha + beta)(2n + alpha + beta + 2))
        B_n = (2n + alpha + beta + 1)(2n + alpha + beta + 2) / (2(n + 1)(n + alpha + beta + 1))
        C_n = (n + alpha)(n + beta)(2n + alpha + beta + 2) / ((n + 1)(n + alpha + beta + 1)(2n + alpha + beta))

    Solving for x * P_n:
        x * P_n = (P_{n+1} - A_n * P_n + C_n * P_{n-1}) / B_n

    Examples
    --------
    >>> a = jacobi_polynomial_p(torch.tensor([1.0]), alpha=0.0, beta=0.0)  # P_0
    >>> b = jacobi_polynomial_p_mulx(a)
    >>> b  # x * P_0^{(0,0)} = P_1^{(0,0)} (for Legendre)
    JacobiPolynomialP(tensor([0., 1.]), alpha=tensor(0.), beta=tensor(0.))
    """
    # Get coefficients as plain tensor
    coeffs = a.as_subclass(torch.Tensor)
    alpha = a.alpha
    beta = a.beta
    n = coeffs.shape[-1]

    ab = alpha + beta

    # Result has one more coefficient
    result_shape = list(coeffs.shape)
    result_shape[-1] = n + 1
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # For each coefficient c_k, we need to express x * P_k^{(alpha,beta)} in the Jacobi basis
    # x * P_k = (P_{k+1} - A_k * P_k + C_k * P_{k-1}) / B_k
    # This means c_k contributes:
    #   c_k / B_k to coefficient of P_{k+1}
    #   -c_k * A_k / B_k to coefficient of P_k
    #   c_k * C_k / B_k to coefficient of P_{k-1}

    for k in range(n):
        c_k = coeffs[..., k]
        k_f = float(k)

        two_k_ab = 2 * k_f + ab

        # Handle special cases for small k or special parameter values
        if k == 0:
            # x * P_0 = ((alpha-beta) + (alpha+beta+2)*x) / 2 in terms of standard normalization
            # For P_0 = 1, P_1 = (alpha - beta)/2 + (alpha + beta + 2)/2 * x
            # So x = (2*P_1 - (alpha - beta)*P_0) / (alpha + beta + 2)
            denom = ab + 2
            if abs(denom.item() if hasattr(denom, "item") else denom) < 1e-15:
                # Edge case: alpha + beta = -2, but we require alpha, beta > -1, so alpha + beta > -2
                # This shouldn't happen, but handle gracefully
                result[..., 1] = result[..., 1] + c_k
            else:
                # x = (2*P_1 - (alpha - beta)*P_0) / (alpha + beta + 2)
                # x * c_0 * P_0 = c_0 * (2*P_1 - (alpha - beta)*P_0) / (alpha + beta + 2)
                result[..., 0] = result[..., 0] - c_k * (alpha - beta) / denom
                result[..., 1] = result[..., 1] + c_k * 2.0 / denom
        else:
            # General case: use the recurrence relation
            # A_k, B_k, C_k for the recurrence
            two_k_ab_p2 = two_k_ab + 2

            # A_k = (alpha^2 - beta^2) / ((2k + alpha + beta)(2k + alpha + beta + 2))
            A_k = (alpha * alpha - beta * beta) / (two_k_ab * two_k_ab_p2)

            # B_k = (2k + alpha + beta + 1)(2k + alpha + beta + 2) / (2(k + 1)(k + alpha + beta + 1))
            B_k = (
                (two_k_ab + 1)
                * two_k_ab_p2
                / (2.0 * (k_f + 1) * (k_f + ab + 1))
            )

            # C_k = (k + alpha)(k + beta)(2k + alpha + beta + 2) / ((k + 1)(k + alpha + beta + 1)(2k + alpha + beta))
            C_k = (
                (k_f + alpha)
                * (k_f + beta)
                * two_k_ab_p2
                / ((k_f + 1) * (k_f + ab + 1) * two_k_ab)
            )

            # x * P_k = (P_{k+1} - A_k * P_k + C_k * P_{k-1}) / B_k
            inv_B_k = 1.0 / B_k

            # Contribution to P_{k-1}
            result[..., k - 1] = result[..., k - 1] + c_k * C_k * inv_B_k
            # Contribution to P_k
            result[..., k] = result[..., k] - c_k * A_k * inv_B_k
            # Contribution to P_{k+1}
            result[..., k + 1] = result[..., k + 1] + c_k * inv_B_k

    return jacobi_polynomial_p(result, alpha.clone(), beta.clone())

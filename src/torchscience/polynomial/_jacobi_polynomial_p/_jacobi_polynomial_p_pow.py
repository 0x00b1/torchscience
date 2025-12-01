import torch

from ._jacobi_polynomial_p import JacobiPolynomialP
from ._jacobi_polynomial_p_multiply import jacobi_polynomial_p_multiply


def jacobi_polynomial_p_pow(
    a: JacobiPolynomialP,
    n: int,
) -> JacobiPolynomialP:
    """Raise Jacobi series to a non-negative integer power.

    Uses binary exponentiation for efficiency.

    Parameters
    ----------
    a : JacobiPolynomialP
        Base series.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    JacobiPolynomialP
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = jacobi_polynomial_p(torch.tensor([1.0, 1.0]), alpha=0.0, beta=0.0)
    >>> b = jacobi_polynomial_p_pow(a, 2)
    >>> # (P_0 + P_1)^2 using Jacobi linearization
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    if n == 0:
        # a^0 = P_0 = 1
        ones_shape = list(a.coeffs.shape)
        ones_shape[-1] = 1
        return JacobiPolynomialP(
            coeffs=torch.ones(
                ones_shape, dtype=a.coeffs.dtype, device=a.coeffs.device
            ),
            alpha=a.alpha.clone(),
            beta=a.beta.clone(),
        )

    if n == 1:
        return JacobiPolynomialP(
            coeffs=a.coeffs.clone(), alpha=a.alpha.clone(), beta=a.beta.clone()
        )

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = JacobiPolynomialP(
                    coeffs=base.coeffs.clone(),
                    alpha=base.alpha.clone(),
                    beta=base.beta.clone(),
                )
            else:
                result = jacobi_polynomial_p_multiply(result, base)
        base = jacobi_polynomial_p_multiply(base, base)
        n //= 2

    return result

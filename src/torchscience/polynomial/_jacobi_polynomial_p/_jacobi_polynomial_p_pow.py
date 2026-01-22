import torch

from ._jacobi_polynomial_p import JacobiPolynomialP, jacobi_polynomial_p
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

    # Get coefficients as plain tensor
    coeffs = a.as_subclass(torch.Tensor)

    if n == 0:
        # a^0 = P_0 = 1
        ones_shape = list(coeffs.shape)
        ones_shape[-1] = 1
        return jacobi_polynomial_p(
            torch.ones(ones_shape, dtype=coeffs.dtype, device=coeffs.device),
            a.alpha.clone(),
            a.beta.clone(),
        )

    if n == 1:
        return jacobi_polynomial_p(
            coeffs.clone(), a.alpha.clone(), a.beta.clone()
        )

    # Binary exponentiation
    result = None
    base = jacobi_polynomial_p(coeffs.clone(), a.alpha.clone(), a.beta.clone())

    while n > 0:
        if n % 2 == 1:
            if result is None:
                base_coeffs = base.as_subclass(torch.Tensor)
                result = jacobi_polynomial_p(
                    base_coeffs.clone(),
                    base.alpha.clone(),
                    base.beta.clone(),
                )
            else:
                result = jacobi_polynomial_p_multiply(result, base)
        base = jacobi_polynomial_p_multiply(base, base)
        n //= 2

    return result

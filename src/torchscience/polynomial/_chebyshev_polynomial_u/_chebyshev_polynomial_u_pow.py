import torch

from ._chebyshev_polynomial_u import ChebyshevPolynomialU
from ._chebyshev_polynomial_u_multiply import chebyshev_polynomial_u_multiply


def chebyshev_polynomial_u_pow(
    a: ChebyshevPolynomialU,
    n: int,
) -> ChebyshevPolynomialU:
    """Raise Chebyshev U series to a non-negative integer power.

    Uses binary exponentiation for efficiency.

    Parameters
    ----------
    a : ChebyshevPolynomialU
        Base series.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    ChebyshevPolynomialU
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = chebyshev_polynomial_u(torch.tensor([1.0, 1.0]))  # 1 + U_1
    >>> b = chebyshev_polynomial_u_pow(a, 2)
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    if n == 0:
        # a^0 = U_0 = 1
        ones_shape = list(a.coeffs.shape)
        ones_shape[-1] = 1
        return ChebyshevPolynomialU(
            coeffs=torch.ones(
                ones_shape, dtype=a.coeffs.dtype, device=a.coeffs.device
            )
        )

    if n == 1:
        return ChebyshevPolynomialU(coeffs=a.coeffs.clone())

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = ChebyshevPolynomialU(coeffs=base.coeffs.clone())
            else:
                result = chebyshev_polynomial_u_multiply(result, base)
        base = chebyshev_polynomial_u_multiply(base, base)
        n //= 2

    return result

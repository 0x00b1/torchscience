import torch

from ._chebyshev_polynomial_t import (
    ChebyshevPolynomialT,
    chebyshev_polynomial_t,
)
from ._chebyshev_polynomial_t_multiply import chebyshev_polynomial_t_multiply


def chebyshev_polynomial_t_pow(
    a: ChebyshevPolynomialT,
    n: int,
) -> ChebyshevPolynomialT:
    """Raise Chebyshev series to a non-negative integer power.

    Uses binary exponentiation for efficiency.

    Parameters
    ----------
    a : ChebyshevPolynomialT
        Base series.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    ChebyshevPolynomialT
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))  # 1 + T_1
    >>> b = chebyshev_polynomial_t_pow(a, 2)
    >>> b  # (1 + T_1)^2 = 1.5 + 2*T_1 + 0.5*T_2
    ChebyshevPolynomialT(tensor([1.5, 2.0, 0.5]))
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    if n == 0:
        # a^0 = T_0 = 1
        ones_shape = list(a.shape)
        ones_shape[-1] = 1
        return chebyshev_polynomial_t(
            torch.ones(ones_shape, dtype=a.dtype, device=a.device)
        )

    if n == 1:
        return chebyshev_polynomial_t(a.clone())

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = chebyshev_polynomial_t(base.clone())
            else:
                result = chebyshev_polynomial_t_multiply(result, base)
        base = chebyshev_polynomial_t_multiply(base, base)
        n //= 2

    return result

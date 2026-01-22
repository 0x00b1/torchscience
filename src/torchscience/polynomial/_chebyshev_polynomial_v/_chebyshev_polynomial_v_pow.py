import torch

from ._chebyshev_polynomial_v import (
    ChebyshevPolynomialV,
    chebyshev_polynomial_v,
)
from ._chebyshev_polynomial_v_multiply import chebyshev_polynomial_v_multiply


def chebyshev_polynomial_v_pow(
    a: ChebyshevPolynomialV,
    n: int,
) -> ChebyshevPolynomialV:
    """Raise Chebyshev V series to a non-negative integer power.

    Uses binary exponentiation for efficiency.

    Parameters
    ----------
    a : ChebyshevPolynomialV
        Base series.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    ChebyshevPolynomialV
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = chebyshev_polynomial_v(torch.tensor([1.0, 1.0]))  # 1 + V_1
    >>> b = chebyshev_polynomial_v_pow(a, 2)
    >>> b  # (1 + V_1)^2 = 1.5 + 2*V_1 + 0.5*V_2
    ChebyshevPolynomialV(tensor([1.5, 2.0, 0.5]))
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    if n == 0:
        # a^0 = V_0 = 1
        ones_shape = list(a.shape)
        ones_shape[-1] = 1
        return chebyshev_polynomial_v(
            torch.ones(ones_shape, dtype=a.dtype, device=a.device)
        )

    if n == 1:
        return chebyshev_polynomial_v(a.clone())

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = chebyshev_polynomial_v(base.clone())
            else:
                result = chebyshev_polynomial_v_multiply(result, base)
        base = chebyshev_polynomial_v_multiply(base, base)
        n //= 2

    return result

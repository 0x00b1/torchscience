import torch
from torch import Tensor

from ._gegenbauer_polynomial_c import (
    GegenbauerPolynomialC,
    gegenbauer_polynomial_c,
)
from ._gegenbauer_polynomial_c_multiply import gegenbauer_polynomial_c_multiply


def gegenbauer_polynomial_c_pow(
    a: GegenbauerPolynomialC,
    n: int,
) -> GegenbauerPolynomialC:
    """Raise Gegenbauer series to a non-negative integer power.

    Uses binary exponentiation for efficiency.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        Base series.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    GegenbauerPolynomialC
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = gegenbauer_polynomial_c(torch.tensor([1.0, 1.0]), torch.tensor(1.0))
    >>> b = gegenbauer_polynomial_c_pow(a, 2)
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    coeffs = a.as_subclass(Tensor)

    if n == 0:
        # a^0 = C_0^{lambda} = 1
        ones_shape = list(coeffs.shape)
        ones_shape[-1] = 1
        return gegenbauer_polynomial_c(
            torch.ones(ones_shape, dtype=coeffs.dtype, device=coeffs.device),
            a.lambda_,
        )

    if n == 1:
        return gegenbauer_polynomial_c(coeffs.clone(), a.lambda_)

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                base_coeffs = base.as_subclass(Tensor)
                result = gegenbauer_polynomial_c(
                    base_coeffs.clone(), base.lambda_
                )
            else:
                result = gegenbauer_polynomial_c_multiply(result, base)
        base = gegenbauer_polynomial_c_multiply(base, base)
        n //= 2

    return result

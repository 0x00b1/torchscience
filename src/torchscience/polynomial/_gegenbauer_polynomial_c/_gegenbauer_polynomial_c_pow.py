import torch

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC
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

    if n == 0:
        # a^0 = C_0^{lambda} = 1
        ones_shape = list(a.coeffs.shape)
        ones_shape[-1] = 1
        return GegenbauerPolynomialC(
            coeffs=torch.ones(
                ones_shape, dtype=a.coeffs.dtype, device=a.coeffs.device
            ),
            lambda_=a.lambda_,
        )

    if n == 1:
        return GegenbauerPolynomialC(
            coeffs=a.coeffs.clone(), lambda_=a.lambda_
        )

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = GegenbauerPolynomialC(
                    coeffs=base.coeffs.clone(), lambda_=base.lambda_
                )
            else:
                result = gegenbauer_polynomial_c_multiply(result, base)
        base = gegenbauer_polynomial_c_multiply(base, base)
        n //= 2

    return result

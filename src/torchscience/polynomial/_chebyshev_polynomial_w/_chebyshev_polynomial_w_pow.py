import torch

from ._chebyshev_polynomial_w import ChebyshevPolynomialW
from ._chebyshev_polynomial_w_multiply import chebyshev_polynomial_w_multiply


def chebyshev_polynomial_w_pow(
    a: ChebyshevPolynomialW,
    n: int,
) -> ChebyshevPolynomialW:
    """Raise Chebyshev W series to a non-negative integer power.

    Uses binary exponentiation for efficiency.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Base series.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    ChebyshevPolynomialW
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = chebyshev_polynomial_w(torch.tensor([1.0, 1.0]))  # 1 + W_1
    >>> b = chebyshev_polynomial_w_pow(a, 2)
    >>> b.coeffs  # (1 + W_1)^2 = 1.5 + 2*W_1 + 0.5*W_2
    tensor([1.5, 2.0, 0.5])
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    if n == 0:
        # a^0 = W_0 = 1
        ones_shape = list(a.coeffs.shape)
        ones_shape[-1] = 1
        return ChebyshevPolynomialW(
            coeffs=torch.ones(
                ones_shape, dtype=a.coeffs.dtype, device=a.coeffs.device
            )
        )

    if n == 1:
        return ChebyshevPolynomialW(coeffs=a.coeffs.clone())

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = ChebyshevPolynomialW(coeffs=base.coeffs.clone())
            else:
                result = chebyshev_polynomial_w_multiply(result, base)
        base = chebyshev_polynomial_w_multiply(base, base)
        n //= 2

    return result

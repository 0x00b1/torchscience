from torch import Tensor

from ._laguerre_polynomial_l import LaguerrePolynomialL


def laguerre_polynomial_l_scale(
    a: LaguerrePolynomialL,
    scalar: Tensor,
) -> LaguerrePolynomialL:
    """Scale a Laguerre series by a scalar.

    Parameters
    ----------
    a : LaguerrePolynomialL
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    LaguerrePolynomialL
        Scaled series scalar * a.

    Examples
    --------
    >>> a = laguerre_polynomial_l(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = laguerre_polynomial_l_scale(a, torch.tensor(2.0))
    >>> b.coeffs
    tensor([2., 4., 6.])
    """
    return LaguerrePolynomialL(coeffs=a.coeffs * scalar)

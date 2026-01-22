import torch

from ._laguerre_polynomial_l import (
    LaguerrePolynomialL,
    laguerre_polynomial_l,
)


def laguerre_polynomial_l_negate(
    a: LaguerrePolynomialL,
) -> LaguerrePolynomialL:
    """Negate a Laguerre series.

    Parameters
    ----------
    a : LaguerrePolynomialL
        Series to negate.

    Returns
    -------
    LaguerrePolynomialL
        Negated series -a.

    Examples
    --------
    >>> a = laguerre_polynomial_l(torch.tensor([1.0, -2.0, 3.0]))
    >>> b = laguerre_polynomial_l_negate(a)
    >>> b
    LaguerrePolynomialL(tensor([-1.,  2., -3.]))
    """
    return laguerre_polynomial_l(-a.as_subclass(torch.Tensor))

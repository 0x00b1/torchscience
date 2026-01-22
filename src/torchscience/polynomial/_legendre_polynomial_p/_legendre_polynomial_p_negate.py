import torch

from ._legendre_polynomial_p import (
    LegendrePolynomialP,
    legendre_polynomial_p,
)


def legendre_polynomial_p_negate(
    a: LegendrePolynomialP,
) -> LegendrePolynomialP:
    """Negate a Legendre series.

    Parameters
    ----------
    a : LegendrePolynomialP
        Series to negate.

    Returns
    -------
    LegendrePolynomialP
        Negated series -a.

    Examples
    --------
    >>> a = legendre_polynomial_p(torch.tensor([1.0, -2.0, 3.0]))
    >>> b = legendre_polynomial_p_negate(a)
    >>> b
    LegendrePolynomialP(tensor([-1.,  2., -3.]))
    """
    return legendre_polynomial_p(-a.as_subclass(torch.Tensor))

from ._legendre_polynomial_p import LegendrePolynomialP


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
    >>> b.coeffs
    tensor([-1.,  2., -3.])
    """
    return LegendrePolynomialP(coeffs=-a.coeffs)

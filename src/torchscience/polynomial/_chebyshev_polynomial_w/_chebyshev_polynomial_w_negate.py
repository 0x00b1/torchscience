from ._chebyshev_polynomial_w import ChebyshevPolynomialW


def chebyshev_polynomial_w_negate(
    a: ChebyshevPolynomialW,
) -> ChebyshevPolynomialW:
    """Negate a Chebyshev W series.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Series to negate.

    Returns
    -------
    ChebyshevPolynomialW
        Negated series -a.

    Examples
    --------
    >>> a = chebyshev_polynomial_w(torch.tensor([1.0, -2.0, 3.0]))
    >>> b = chebyshev_polynomial_w_negate(a)
    >>> b.coeffs
    tensor([-1.,  2., -3.])
    """
    return ChebyshevPolynomialW(coeffs=-a.coeffs)

from ._chebyshev_polynomial_v import ChebyshevPolynomialV


def chebyshev_polynomial_v_negate(
    a: ChebyshevPolynomialV,
) -> ChebyshevPolynomialV:
    """Negate a Chebyshev V series.

    Parameters
    ----------
    a : ChebyshevPolynomialV
        Series to negate.

    Returns
    -------
    ChebyshevPolynomialV
        Negated series -a.

    Examples
    --------
    >>> a = chebyshev_polynomial_v(torch.tensor([1.0, -2.0, 3.0]))
    >>> b = chebyshev_polynomial_v_negate(a)
    >>> b.coeffs
    tensor([-1.,  2., -3.])
    """
    return ChebyshevPolynomialV(coeffs=-a.coeffs)

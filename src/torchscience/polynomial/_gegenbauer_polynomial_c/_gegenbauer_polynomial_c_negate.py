from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_negate(
    a: GegenbauerPolynomialC,
) -> GegenbauerPolynomialC:
    """Negate a Gegenbauer series.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        Series to negate.

    Returns
    -------
    GegenbauerPolynomialC
        Negated series -a.

    Examples
    --------
    >>> a = gegenbauer_polynomial_c(torch.tensor([1.0, -2.0, 3.0]), torch.tensor(1.0))
    >>> b = gegenbauer_polynomial_c_negate(a)
    >>> b.coeffs
    tensor([-1.,  2., -3.])
    """
    return GegenbauerPolynomialC(coeffs=-a.coeffs, lambda_=a.lambda_)

from ._laguerre_polynomial_l import LaguerrePolynomialL


def laguerre_polynomial_l_degree(
    p: LaguerrePolynomialL,
) -> int:
    """Return degree of Laguerre series.

    Parameters
    ----------
    p : LaguerrePolynomialL
        Laguerre series.

    Returns
    -------
    int
        Degree of the polynomial (number of coefficients - 1).

    Notes
    -----
    The degree is determined by the number of stored coefficients,
    not by the actual polynomial degree after removing trailing zeros.
    Use `laguerre_polynomial_l_trim` first if you want the true degree.

    Examples
    --------
    >>> c = laguerre_polynomial_l(torch.tensor([1.0, 2.0, 3.0]))
    >>> laguerre_polynomial_l_degree(c)
    2
    """
    return p.shape[-1] - 1

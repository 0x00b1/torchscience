from ._jacobi_polynomial_p import JacobiPolynomialP
from ._jacobi_polynomial_p_divmod import jacobi_polynomial_p_divmod


def jacobi_polynomial_p_mod(
    a: JacobiPolynomialP,
    b: JacobiPolynomialP,
) -> JacobiPolynomialP:
    """Divide two Jacobi series, returning remainder only.

    Parameters
    ----------
    a : JacobiPolynomialP
        Dividend.
    b : JacobiPolynomialP
        Divisor.

    Returns
    -------
    JacobiPolynomialP
        Remainder.

    Raises
    ------
    ParameterMismatchError
        If the series have different alpha or beta parameters.
    """
    _, r = jacobi_polynomial_p_divmod(a, b)
    return r

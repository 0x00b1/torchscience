from ._jacobi_polynomial_p import JacobiPolynomialP
from ._jacobi_polynomial_p_divmod import jacobi_polynomial_p_divmod


def jacobi_polynomial_p_div(
    a: JacobiPolynomialP,
    b: JacobiPolynomialP,
) -> JacobiPolynomialP:
    """Divide two Jacobi series, returning quotient only.

    Parameters
    ----------
    a : JacobiPolynomialP
        Dividend.
    b : JacobiPolynomialP
        Divisor.

    Returns
    -------
    JacobiPolynomialP
        Quotient.

    Raises
    ------
    ParameterMismatchError
        If the series have different alpha or beta parameters.
    """
    q, _ = jacobi_polynomial_p_divmod(a, b)
    return q

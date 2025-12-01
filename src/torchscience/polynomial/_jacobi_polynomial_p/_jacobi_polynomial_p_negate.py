from ._jacobi_polynomial_p import JacobiPolynomialP


def jacobi_polynomial_p_negate(
    a: JacobiPolynomialP,
) -> JacobiPolynomialP:
    """Negate a Jacobi series.

    Parameters
    ----------
    a : JacobiPolynomialP
        Series to negate.

    Returns
    -------
    JacobiPolynomialP
        Negated series -a.

    Examples
    --------
    >>> a = jacobi_polynomial_p(torch.tensor([1.0, -2.0, 3.0]), alpha=0.5, beta=0.5)
    >>> b = jacobi_polynomial_p_negate(a)
    >>> b.coeffs
    tensor([-1.,  2., -3.])
    """
    return JacobiPolynomialP(
        coeffs=-a.coeffs, alpha=a.alpha.clone(), beta=a.beta.clone()
    )

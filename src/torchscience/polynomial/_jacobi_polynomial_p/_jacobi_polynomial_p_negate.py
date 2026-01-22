import torch

from ._jacobi_polynomial_p import JacobiPolynomialP, jacobi_polynomial_p


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
    >>> b
    JacobiPolynomialP(tensor([-1.,  2., -3.]), alpha=tensor(0.5000), beta=tensor(0.5000))
    """
    # Get coefficients as plain tensor
    coeffs = a.as_subclass(torch.Tensor)
    return jacobi_polynomial_p(-coeffs, a.alpha.clone(), a.beta.clone())

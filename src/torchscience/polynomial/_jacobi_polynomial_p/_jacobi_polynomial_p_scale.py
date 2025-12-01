from torch import Tensor

from ._jacobi_polynomial_p import JacobiPolynomialP


def jacobi_polynomial_p_scale(
    a: JacobiPolynomialP,
    scalar: Tensor,
) -> JacobiPolynomialP:
    """Scale a Jacobi series by a scalar.

    Parameters
    ----------
    a : JacobiPolynomialP
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    JacobiPolynomialP
        Scaled series scalar * a.

    Examples
    --------
    >>> a = jacobi_polynomial_p(torch.tensor([1.0, 2.0, 3.0]), alpha=0.5, beta=0.5)
    >>> b = jacobi_polynomial_p_scale(a, torch.tensor(2.0))
    >>> b.coeffs
    tensor([2., 4., 6.])
    """
    return JacobiPolynomialP(
        coeffs=a.coeffs * scalar, alpha=a.alpha.clone(), beta=a.beta.clone()
    )

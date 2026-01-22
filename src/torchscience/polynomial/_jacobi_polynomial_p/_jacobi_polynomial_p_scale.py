import torch
from torch import Tensor

from ._jacobi_polynomial_p import JacobiPolynomialP, jacobi_polynomial_p


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
    >>> b
    JacobiPolynomialP(tensor([2., 4., 6.]), alpha=tensor(0.5000), beta=tensor(0.5000))
    """
    # Get coefficients as plain tensor
    coeffs = a.as_subclass(torch.Tensor)
    return jacobi_polynomial_p(
        coeffs * scalar, a.alpha.clone(), a.beta.clone()
    )

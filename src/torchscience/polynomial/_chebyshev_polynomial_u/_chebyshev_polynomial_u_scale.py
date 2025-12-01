from torch import Tensor

from ._chebyshev_polynomial_u import ChebyshevPolynomialU


def chebyshev_polynomial_u_scale(
    a: ChebyshevPolynomialU,
    scalar: Tensor,
) -> ChebyshevPolynomialU:
    """Scale a Chebyshev U series by a scalar.

    Parameters
    ----------
    a : ChebyshevPolynomialU
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    ChebyshevPolynomialU
        Scaled series scalar * a.

    Examples
    --------
    >>> a = chebyshev_polynomial_u(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = chebyshev_polynomial_u_scale(a, torch.tensor(2.0))
    >>> b.coeffs
    tensor([2., 4., 6.])
    """
    return ChebyshevPolynomialU(coeffs=a.coeffs * scalar)

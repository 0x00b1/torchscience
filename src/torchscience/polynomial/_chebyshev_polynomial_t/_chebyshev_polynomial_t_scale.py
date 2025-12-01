from torch import Tensor

from ._chebyshev_polynomial_t import ChebyshevPolynomialT


def chebyshev_polynomial_t_scale(
    a: ChebyshevPolynomialT,
    scalar: Tensor,
) -> ChebyshevPolynomialT:
    """Scale a Chebyshev series by a scalar.

    Parameters
    ----------
    a : ChebyshevPolynomialT
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    ChebyshevPolynomialT
        Scaled series scalar * a.

    Examples
    --------
    >>> a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = chebyshev_polynomial_t_scale(a, torch.tensor(2.0))
    >>> b.coeffs
    tensor([2., 4., 6.])
    """
    return ChebyshevPolynomialT(coeffs=a.coeffs * scalar)

from torch import Tensor

from ._hermite_polynomial_h import HermitePolynomialH


def hermite_polynomial_h_scale(
    a: HermitePolynomialH,
    scalar: Tensor,
) -> HermitePolynomialH:
    """Scale a Physicists' Hermite series by a scalar.

    Parameters
    ----------
    a : HermitePolynomialH
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    HermitePolynomialH
        Scaled series scalar * a.

    Examples
    --------
    >>> a = hermite_polynomial_h(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = hermite_polynomial_h_scale(a, torch.tensor(2.0))
    >>> b.coeffs
    tensor([2., 4., 6.])
    """
    return HermitePolynomialH(coeffs=a.coeffs * scalar)

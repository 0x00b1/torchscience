from torch import Tensor

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_scale(
    a: GegenbauerPolynomialC,
    scalar: Tensor,
) -> GegenbauerPolynomialC:
    """Scale a Gegenbauer series by a scalar.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    GegenbauerPolynomialC
        Scaled series scalar * a.

    Examples
    --------
    >>> a = gegenbauer_polynomial_c(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(1.0))
    >>> b = gegenbauer_polynomial_c_scale(a, torch.tensor(2.0))
    >>> b.coeffs
    tensor([2., 4., 6.])
    """
    return GegenbauerPolynomialC(coeffs=a.coeffs * scalar, lambda_=a.lambda_)

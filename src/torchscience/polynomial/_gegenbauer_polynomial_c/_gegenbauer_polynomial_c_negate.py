from torch import Tensor

from ._gegenbauer_polynomial_c import (
    GegenbauerPolynomialC,
    gegenbauer_polynomial_c,
)


def gegenbauer_polynomial_c_negate(
    a: GegenbauerPolynomialC,
) -> GegenbauerPolynomialC:
    """Negate a Gegenbauer series.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        Series to negate.

    Returns
    -------
    GegenbauerPolynomialC
        Negated series -a.

    Examples
    --------
    >>> a = gegenbauer_polynomial_c(torch.tensor([1.0, -2.0, 3.0]), torch.tensor(1.0))
    >>> b = gegenbauer_polynomial_c_negate(a)
    >>> b
    GegenbauerPolynomialC(tensor([-1.,  2., -3.]), lambda_=tensor(1.))
    """
    coeffs = a.as_subclass(Tensor)
    return gegenbauer_polynomial_c(-coeffs, a.lambda_)

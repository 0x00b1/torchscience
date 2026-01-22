import torch

from ._chebyshev_polynomial_w import (
    ChebyshevPolynomialW,
    chebyshev_polynomial_w,
)


def chebyshev_polynomial_w_negate(
    a: ChebyshevPolynomialW,
) -> ChebyshevPolynomialW:
    """Negate a Chebyshev W series.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Series to negate.

    Returns
    -------
    ChebyshevPolynomialW
        Negated series -a.

    Examples
    --------
    >>> a = chebyshev_polynomial_w(torch.tensor([1.0, -2.0, 3.0]))
    >>> b = chebyshev_polynomial_w_negate(a)
    >>> b
    ChebyshevPolynomialW(tensor([-1.,  2., -3.]))
    """
    result = torch.Tensor.neg(a)
    return chebyshev_polynomial_w(result)

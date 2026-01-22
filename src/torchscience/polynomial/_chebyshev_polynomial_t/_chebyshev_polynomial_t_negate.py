import torch

from ._chebyshev_polynomial_t import (
    ChebyshevPolynomialT,
    chebyshev_polynomial_t,
)


def chebyshev_polynomial_t_negate(
    a: ChebyshevPolynomialT,
) -> ChebyshevPolynomialT:
    """Negate a Chebyshev series.

    Parameters
    ----------
    a : ChebyshevPolynomialT
        Series to negate.

    Returns
    -------
    ChebyshevPolynomialT
        Negated series -a.

    Examples
    --------
    >>> a = chebyshev_polynomial_t(torch.tensor([1.0, -2.0, 3.0]))
    >>> b = chebyshev_polynomial_t_negate(a)
    >>> b
    ChebyshevPolynomialT(tensor([-1.,  2., -3.]))
    """
    return chebyshev_polynomial_t(-a.as_subclass(torch.Tensor))

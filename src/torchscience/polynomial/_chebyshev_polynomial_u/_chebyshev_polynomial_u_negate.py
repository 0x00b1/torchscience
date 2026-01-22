import torch

from ._chebyshev_polynomial_u import (
    ChebyshevPolynomialU,
    chebyshev_polynomial_u,
)


def chebyshev_polynomial_u_negate(
    a: ChebyshevPolynomialU,
) -> ChebyshevPolynomialU:
    """Negate a Chebyshev U series.

    Parameters
    ----------
    a : ChebyshevPolynomialU
        Series to negate.

    Returns
    -------
    ChebyshevPolynomialU
        Negated series -a.

    Examples
    --------
    >>> a = chebyshev_polynomial_u(torch.tensor([1.0, -2.0, 3.0]))
    >>> b = chebyshev_polynomial_u_negate(a)
    >>> b
    ChebyshevPolynomialU(tensor([-1.,  2., -3.]))
    """
    return chebyshev_polynomial_u(-a.as_subclass(torch.Tensor))

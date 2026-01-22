import torch
from torch import Tensor

from ._chebyshev_polynomial_u import ChebyshevPolynomialU


def chebyshev_polynomial_u_degree(
    c: ChebyshevPolynomialU,
) -> Tensor:
    """Return degree of Chebyshev U series.

    Parameters
    ----------
    c : ChebyshevPolynomialU
        Chebyshev U series.

    Returns
    -------
    Tensor
        Degree (number of coefficients - 1).

    Examples
    --------
    >>> c = chebyshev_polynomial_u(torch.tensor([1.0, 2.0, 3.0]))
    >>> chebyshev_polynomial_u_degree(c)
    tensor(2)
    """
    return torch.tensor(c.shape[-1] - 1)

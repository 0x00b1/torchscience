import torch
from torch import Tensor

from ._chebyshev_polynomial_t import ChebyshevPolynomialT


def chebyshev_polynomial_t_degree(
    c: ChebyshevPolynomialT,
) -> Tensor:
    """Return degree of Chebyshev series.

    Parameters
    ----------
    c : ChebyshevPolynomialT
        Chebyshev series.

    Returns
    -------
    Tensor
        Degree (number of coefficients - 1).

    Examples
    --------
    >>> c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
    >>> chebyshev_polynomial_t_degree(c)
    tensor(2)
    """
    return torch.tensor(c.shape[-1] - 1)

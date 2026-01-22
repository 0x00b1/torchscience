import torch
from torch import Tensor

from ._chebyshev_polynomial_v import ChebyshevPolynomialV


def chebyshev_polynomial_v_degree(
    c: ChebyshevPolynomialV,
) -> Tensor:
    """Return degree of Chebyshev V series.

    Parameters
    ----------
    c : ChebyshevPolynomialV
        Chebyshev V series.

    Returns
    -------
    Tensor
        Degree (number of coefficients - 1).

    Examples
    --------
    >>> c = chebyshev_polynomial_v(torch.tensor([1.0, 2.0, 3.0]))
    >>> chebyshev_polynomial_v_degree(c)
    tensor(2)
    """
    return torch.tensor(c.shape[-1] - 1)

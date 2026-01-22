import torch
from torch import Tensor

from ._chebyshev_polynomial_w import ChebyshevPolynomialW


def chebyshev_polynomial_w_degree(
    a: ChebyshevPolynomialW,
) -> Tensor:
    """Return the degree of the Chebyshev W series.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Input series.

    Returns
    -------
    Tensor
        Degree (number of coefficients minus 1).

    Notes
    -----
    This returns the formal degree based on coefficient count,
    not the true degree (which would require checking for zeros).

    Examples
    --------
    >>> a = chebyshev_polynomial_w(torch.tensor([1.0, 2.0, 3.0]))
    >>> chebyshev_polynomial_w_degree(a)
    tensor(2)
    """
    return torch.tensor(a.shape[-1] - 1)

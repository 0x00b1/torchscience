import torch
from torch import Tensor

from ._legendre_polynomial_p import LegendrePolynomialP


def legendre_polynomial_p_degree(p: LegendrePolynomialP) -> Tensor:
    """Return degree of Legendre polynomial series.

    Parameters
    ----------
    p : LegendrePolynomialP
        Input Legendre series.

    Returns
    -------
    Tensor
        Degree, shape matches batch dimensions.
        Returns number of coefficients minus 1.

    Notes
    -----
    This returns the formal degree (len(coeffs) - 1), not the actual degree
    which would require checking for trailing zeros. Use legendre_polynomial_p_trim
    first if you need the actual degree.

    Examples
    --------
    >>> c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
    >>> legendre_polynomial_p_degree(c)
    tensor(2)
    """
    return torch.tensor(p.shape[-1] - 1, device=p.device)

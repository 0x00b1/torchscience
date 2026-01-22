import torch
from torch import Tensor

from ._hermite_polynomial_he import HermitePolynomialHe


def hermite_polynomial_he_degree(p: HermitePolynomialHe) -> Tensor:
    """Return degree of Probabilists' Hermite polynomial series.

    Parameters
    ----------
    p : HermitePolynomialHe
        Input Hermite series.

    Returns
    -------
    Tensor
        Degree, shape matches batch dimensions.
        Returns number of coefficients minus 1.

    Notes
    -----
    This returns the formal degree (len(coeffs) - 1), not the actual degree
    which would require checking for trailing zeros. Use hermite_polynomial_he_trim
    first if you need the actual degree.

    Examples
    --------
    >>> c = hermite_polynomial_he(torch.tensor([1.0, 2.0, 3.0]))
    >>> hermite_polynomial_he_degree(c)
    tensor(2)
    """
    return torch.tensor(p.shape[-1] - 1, device=p.device)

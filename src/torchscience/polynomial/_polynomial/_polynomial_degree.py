import torch
from torch import Tensor

from ._polynomial import Polynomial


def polynomial_degree(p: Polynomial) -> Tensor:
    """Return degree of polynomial(s).

    Parameters
    ----------
    p : Polynomial
        Input polynomial.

    Returns
    -------
    Tensor
        Degree, shape matches batch dimensions.
        Returns number of coefficients minus 1.

    Notes
    -----
    This returns the formal degree (len(coeffs) - 1), not the actual degree
    which would require checking for trailing zeros. Use polynomial_trim
    first if you need the actual degree.
    """
    return torch.tensor(p.coeffs.shape[-1] - 1, device=p.coeffs.device)

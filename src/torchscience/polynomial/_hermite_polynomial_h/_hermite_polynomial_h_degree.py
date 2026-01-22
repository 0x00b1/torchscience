import torch
from torch import Tensor

from ._hermite_polynomial_h import HermitePolynomialH


def hermite_polynomial_h_degree(p: HermitePolynomialH) -> Tensor:
    """Return degree of Physicists' Hermite polynomial series.

    Parameters
    ----------
    p : HermitePolynomialH
        Input Hermite series.

    Returns
    -------
    Tensor
        Degree, shape matches batch dimensions.
        Returns number of coefficients minus 1.

    Notes
    -----
    This returns the formal degree (len(coeffs) - 1), not the actual degree
    which would require checking for trailing zeros. Use hermite_polynomial_h_trim
    first if you need the actual degree.

    Examples
    --------
    >>> c = hermite_polynomial_h(torch.tensor([1.0, 2.0, 3.0]))
    >>> hermite_polynomial_h_degree(c)
    tensor(2)
    """
    return torch.tensor(p.shape[-1] - 1, device=p.device)

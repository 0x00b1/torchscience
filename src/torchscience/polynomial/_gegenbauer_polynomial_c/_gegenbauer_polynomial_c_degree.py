import torch
from torch import Tensor

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_degree(p: GegenbauerPolynomialC) -> Tensor:
    """Return degree of Gegenbauer polynomial series.

    Parameters
    ----------
    p : GegenbauerPolynomialC
        Input Gegenbauer series.

    Returns
    -------
    Tensor
        Degree, shape matches batch dimensions.
        Returns number of coefficients minus 1.

    Notes
    -----
    This returns the formal degree (len(coeffs) - 1), not the actual degree
    which would require checking for trailing zeros. Use gegenbauer_polynomial_c_trim
    first if you need the actual degree.

    Examples
    --------
    >>> c = gegenbauer_polynomial_c(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(1.0))
    >>> gegenbauer_polynomial_c_degree(c)
    tensor(2)
    """
    return torch.tensor(p.coeffs.shape[-1] - 1, device=p.coeffs.device)

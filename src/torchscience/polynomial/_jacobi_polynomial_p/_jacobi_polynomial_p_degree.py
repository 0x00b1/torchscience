import torch
from torch import Tensor

from ._jacobi_polynomial_p import JacobiPolynomialP


def jacobi_polynomial_p_degree(p: JacobiPolynomialP) -> Tensor:
    """Return degree of Jacobi polynomial series.

    Parameters
    ----------
    p : JacobiPolynomialP
        Input Jacobi series.

    Returns
    -------
    Tensor
        Degree, shape matches batch dimensions.
        Returns number of coefficients minus 1.

    Notes
    -----
    This returns the formal degree (len(coeffs) - 1), not the actual degree
    which would require checking for trailing zeros. Use jacobi_polynomial_p_trim
    first if you need the actual degree.

    Examples
    --------
    >>> c = jacobi_polynomial_p(torch.tensor([1.0, 2.0, 3.0]), alpha=0.0, beta=0.0)
    >>> jacobi_polynomial_p_degree(c)
    tensor(2)
    """
    return torch.tensor(p.coeffs.shape[-1] - 1, device=p.coeffs.device)

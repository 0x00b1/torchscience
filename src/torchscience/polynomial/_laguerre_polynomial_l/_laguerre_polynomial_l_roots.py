import torch
from torch import Tensor

from ._laguerre_polynomial_l import LaguerrePolynomialL
from ._laguerre_polynomial_l_companion import laguerre_polynomial_l_companion


def laguerre_polynomial_l_roots(
    c: LaguerrePolynomialL,
) -> Tensor:
    """Find roots of Laguerre series.

    Computes roots as eigenvalues of the companion matrix.

    Parameters
    ----------
    c : LaguerrePolynomialL
        Laguerre series.

    Returns
    -------
    Tensor
        Complex tensor of roots, shape (n,) where n = degree.

    Notes
    -----
    Uses eigenvalue decomposition of the Laguerre companion matrix.
    Roots are returned as complex numbers even if all roots are real.

    For the nth Laguerre polynomial L_n(x), the roots are the
    Gauss-Laguerre quadrature points, which are all positive real numbers.

    Examples
    --------
    >>> c = laguerre_polynomial_l(torch.tensor([0.0, 0.0, 1.0]))  # L_2
    >>> roots = laguerre_polynomial_l_roots(c)
    >>> roots.real.sort().values
    tensor([0.5858, 3.4142])
    """
    A = laguerre_polynomial_l_companion(c)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(A)

    return eigenvalues

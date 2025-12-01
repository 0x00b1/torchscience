import torch
from torch import Tensor

from ._legendre_polynomial_p import LegendrePolynomialP
from ._legendre_polynomial_p_companion import legendre_polynomial_p_companion


def legendre_polynomial_p_roots(
    c: LegendrePolynomialP,
) -> Tensor:
    """Find roots of Legendre series.

    Computes roots as eigenvalues of the companion matrix.

    Parameters
    ----------
    c : LegendrePolynomialP
        Legendre series.

    Returns
    -------
    Tensor
        Complex tensor of roots, shape (n,) where n = degree.

    Notes
    -----
    Uses eigenvalue decomposition of the Legendre companion matrix.
    Roots are returned as complex numbers even if all roots are real.

    For the nth Legendre polynomial P_n(x), the roots are the
    Gauss-Legendre quadrature points, which lie in (-1, 1).

    Examples
    --------
    >>> c = legendre_polynomial_p(torch.tensor([0.0, 0.0, 1.0]))  # P_2
    >>> roots = legendre_polynomial_p_roots(c)
    >>> roots.real.sort().values
    tensor([-0.5774,  0.5774])
    """
    A = legendre_polynomial_p_companion(c)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(A)

    return eigenvalues

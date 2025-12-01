import torch
from torch import Tensor

from ._chebyshev_polynomial_u import ChebyshevPolynomialU
from ._chebyshev_polynomial_u_companion import chebyshev_polynomial_u_companion


def chebyshev_polynomial_u_roots(
    c: ChebyshevPolynomialU,
) -> Tensor:
    """Find roots of Chebyshev U series.

    Computes roots as eigenvalues of the companion matrix.

    Parameters
    ----------
    c : ChebyshevPolynomialU
        Chebyshev U series.

    Returns
    -------
    Tensor
        Complex tensor of roots, shape (n,) where n = degree.

    Notes
    -----
    Uses eigenvalue decomposition of the Chebyshev U companion matrix.
    Roots are returned as complex numbers even if all roots are real.

    Examples
    --------
    >>> c = chebyshev_polynomial_u(torch.tensor([0.0, 0.0, 1.0]))  # U_2
    >>> roots = chebyshev_polynomial_u_roots(c)
    >>> roots.real.sort().values
    tensor([-0.5000,  0.5000])
    """
    A = chebyshev_polynomial_u_companion(c)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(A)

    return eigenvalues

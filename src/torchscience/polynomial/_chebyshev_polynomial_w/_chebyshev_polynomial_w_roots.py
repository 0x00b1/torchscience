import torch
from torch import Tensor

from ._chebyshev_polynomial_w import ChebyshevPolynomialW
from ._chebyshev_polynomial_w_companion import chebyshev_polynomial_w_companion


def chebyshev_polynomial_w_roots(
    c: ChebyshevPolynomialW,
) -> Tensor:
    """Find roots of Chebyshev W series.

    Computes roots as eigenvalues of the companion matrix.

    Parameters
    ----------
    c : ChebyshevPolynomialW
        Chebyshev W series.

    Returns
    -------
    Tensor
        Complex tensor of roots, shape (n,) where n = degree.

    Notes
    -----
    Uses eigenvalue decomposition of the Chebyshev W companion matrix.
    Roots are returned as complex numbers even if all roots are real.

    Examples
    --------
    >>> c = ChebyshevPolynomialW(coeffs=torch.tensor([0.0, 0.0, 1.0]))  # W_2
    >>> roots = chebyshev_polynomial_w_roots(c)
    """
    A = chebyshev_polynomial_w_companion(c)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(A)

    return eigenvalues

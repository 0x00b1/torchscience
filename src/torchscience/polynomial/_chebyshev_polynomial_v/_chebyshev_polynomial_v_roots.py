import torch
from torch import Tensor

from ._chebyshev_polynomial_v import ChebyshevPolynomialV
from ._chebyshev_polynomial_v_companion import chebyshev_polynomial_v_companion


def chebyshev_polynomial_v_roots(
    c: ChebyshevPolynomialV,
) -> Tensor:
    """Find roots of Chebyshev V series.

    Computes roots as eigenvalues of the companion matrix.

    Parameters
    ----------
    c : ChebyshevPolynomialV
        Chebyshev V series.

    Returns
    -------
    Tensor
        Complex tensor of roots, shape (n,) where n = degree.

    Notes
    -----
    Uses eigenvalue decomposition of the Chebyshev V companion matrix.
    Roots are returned as complex numbers even if all roots are real.

    Examples
    --------
    >>> c = chebyshev_polynomial_v(torch.tensor([0.0, 0.0, 1.0]))  # V_2
    >>> roots = chebyshev_polynomial_v_roots(c)
    """
    A = chebyshev_polynomial_v_companion(c)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(A)

    return eigenvalues

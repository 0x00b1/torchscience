import torch
from torch import Tensor

from ._chebyshev_polynomial_t import ChebyshevPolynomialT
from ._chebyshev_polynomial_t_companion import chebyshev_polynomial_t_companion


def chebyshev_polynomial_t_roots(
    c: ChebyshevPolynomialT,
) -> Tensor:
    """Find roots of Chebyshev series.

    Computes roots as eigenvalues of the companion matrix.

    Parameters
    ----------
    c : ChebyshevPolynomialT
        Chebyshev series.

    Returns
    -------
    Tensor
        Complex tensor of roots, shape (n,) where n = degree.

    Notes
    -----
    Uses eigenvalue decomposition of the Chebyshev companion matrix.
    Roots are returned as complex numbers even if all roots are real.

    Examples
    --------
    >>> c = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
    >>> roots = chebyshev_polynomial_t_roots(c)
    >>> roots.real.sort().values
    tensor([-0.7071,  0.7071])
    """
    return torch.linalg.eigvals(chebyshev_polynomial_t_companion(c))

import torch
from torch import Tensor

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC
from ._gegenbauer_polynomial_c_companion import (
    gegenbauer_polynomial_c_companion,
)


def gegenbauer_polynomial_c_roots(
    c: GegenbauerPolynomialC,
) -> Tensor:
    """Find roots of Gegenbauer series.

    Computes roots as eigenvalues of the companion matrix.

    Parameters
    ----------
    c : GegenbauerPolynomialC
        Gegenbauer series.

    Returns
    -------
    Tensor
        Complex tensor of roots, shape (n,) where n = degree.

    Notes
    -----
    Uses eigenvalue decomposition of the companion matrix.
    Roots are returned as complex numbers even if all roots are real.

    For the nth Gegenbauer polynomial C_n^{lambda}(x), the roots are the
    Gauss-Gegenbauer quadrature points, which lie in (-1, 1).

    Examples
    --------
    >>> c = gegenbauer_polynomial_c(
    ...     torch.tensor([0.0, 0.0, 1.0]), torch.tensor(1.0)
    ... )  # C_2^1
    >>> roots = gegenbauer_polynomial_c_roots(c)
    >>> roots.real.sort().values
    tensor([-0.5774,  0.5774])
    """
    A = gegenbauer_polynomial_c_companion(c)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(A)

    return eigenvalues

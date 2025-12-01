import torch
from torch import Tensor

from ._jacobi_polynomial_p import JacobiPolynomialP
from ._jacobi_polynomial_p_companion import jacobi_polynomial_p_companion


def jacobi_polynomial_p_roots(
    c: JacobiPolynomialP,
) -> Tensor:
    """Find roots of Jacobi series.

    Computes roots as eigenvalues of the companion matrix.

    Parameters
    ----------
    c : JacobiPolynomialP
        Jacobi series.

    Returns
    -------
    Tensor
        Complex tensor of roots, shape (n,) where n = degree.

    Notes
    -----
    Uses eigenvalue decomposition of the Jacobi companion matrix.
    Roots are returned as complex numbers even if all roots are real.

    For the nth Jacobi polynomial P_n^{(α,β)}(x), the roots are the
    Gauss-Jacobi quadrature points, which lie in (-1, 1).

    Examples
    --------
    >>> c = jacobi_polynomial_p(torch.tensor([0.0, 0.0, 1.0]), alpha=0.0, beta=0.0)
    >>> roots = jacobi_polynomial_p_roots(c)
    >>> roots.real.sort().values
    tensor([-0.5774,  0.5774])
    """
    A = jacobi_polynomial_p_companion(c)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(A)

    return eigenvalues

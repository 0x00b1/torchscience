import numpy as np
import torch
from torch import Tensor

from ._hermite_polynomial_he import HermitePolynomialHe


def hermite_polynomial_he_roots(
    c: HermitePolynomialHe,
) -> Tensor:
    """Find roots of Probabilists' Hermite series.

    Computes roots using NumPy's hermeroots for numerical stability.

    Parameters
    ----------
    c : HermitePolynomialHe
        Hermite series.

    Returns
    -------
    Tensor
        Complex tensor of roots, shape (n,) where n = degree.

    Notes
    -----
    Uses NumPy's hermeroots which implements eigenvalue decomposition
    of the companion matrix with proper scaling for numerical stability.

    Roots are returned as complex numbers even if all roots are real.

    For the nth Hermite_e polynomial He_n(x), the roots are the
    Gauss-Hermite_e quadrature points.

    Examples
    --------
    >>> c = hermite_polynomial_he(torch.tensor([0.0, 0.0, 1.0]))  # He_2
    >>> roots = hermite_polynomial_he_roots(c)
    >>> roots.real.sort().values  # He_2 = x^2 - 1 has roots at +/- 1
    tensor([-1.,  1.])
    """
    # Convert to plain tensor for operations
    coeffs = c.as_subclass(torch.Tensor)

    # Use NumPy's hermeroots for numerical stability
    coeffs_np = coeffs.detach().cpu().numpy()
    roots_np = np.polynomial.hermite_e.hermeroots(coeffs_np)

    # Convert to complex tensor
    roots = torch.from_numpy(roots_np.astype(np.complex128)).to(
        device=coeffs.device
    )

    return roots

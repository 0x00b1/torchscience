import numpy as np
import torch
from torch import Tensor

from ._hermite_polynomial_h import HermitePolynomialH


def hermite_polynomial_h_roots(
    c: HermitePolynomialH,
) -> Tensor:
    """Find roots of Physicists' Hermite series.

    Computes roots using NumPy's hermroots for numerical stability.

    Parameters
    ----------
    c : HermitePolynomialH
        Hermite series.

    Returns
    -------
    Tensor
        Complex tensor of roots, shape (n,) where n = degree.

    Notes
    -----
    Uses NumPy's hermroots which implements eigenvalue decomposition
    of the companion matrix with proper scaling for numerical stability.

    Roots are returned as complex numbers even if all roots are real.

    For the nth Hermite polynomial H_n(x), the roots are the
    Gauss-Hermite quadrature points.

    Examples
    --------
    >>> c = hermite_polynomial_h(torch.tensor([0.0, 0.0, 1.0]))  # H_2
    >>> roots = hermite_polynomial_h_roots(c)
    >>> roots.real.sort().values  # H_2 = 4x^2 - 2 has roots at +/- 1/sqrt(2)
    tensor([-0.7071,  0.7071])
    """
    coeffs = c.coeffs

    # Use NumPy's hermroots for numerical stability
    coeffs_np = coeffs.detach().cpu().numpy()
    roots_np = np.polynomial.hermite.hermroots(coeffs_np)

    # Convert to complex tensor
    roots = torch.from_numpy(roots_np.astype(np.complex128)).to(
        device=coeffs.device
    )

    return roots

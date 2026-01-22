import numpy as np
import torch
from torch import Tensor

from ._hermite_polynomial_h import (
    HermitePolynomialH,
    hermite_polynomial_h,
)


def hermite_polynomial_h_from_roots(
    roots: Tensor,
) -> HermitePolynomialH:
    """Construct monic Physicists' Hermite series from its roots.

    The resulting series is (x - r_0)(x - r_1)...(x - r_{n-1}) expressed
    in the Hermite basis.

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial, shape (n,).

    Returns
    -------
    HermitePolynomialH
        Monic Hermite series with the given roots.

    Notes
    -----
    Uses NumPy's hermfromroots which constructs the polynomial from
    roots in the Hermite basis.

    Examples
    --------
    >>> roots = torch.tensor([0.5, -0.5])
    >>> c = hermite_polynomial_h_from_roots(roots)
    """
    # Use NumPy's hermfromroots
    roots_np = roots.detach().cpu().numpy()
    coeffs_np = np.polynomial.hermite.hermfromroots(roots_np)

    coeffs = torch.from_numpy(coeffs_np).to(
        dtype=roots.dtype if roots.is_floating_point() else torch.float32,
        device=roots.device,
    )

    return hermite_polynomial_h(coeffs)

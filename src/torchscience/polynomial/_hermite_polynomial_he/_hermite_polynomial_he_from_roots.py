import numpy as np
import torch
from torch import Tensor

from ._hermite_polynomial_he import HermitePolynomialHe


def hermite_polynomial_he_from_roots(
    roots: Tensor,
) -> HermitePolynomialHe:
    """Construct monic Probabilists' Hermite series from its roots.

    The resulting series is (x - r_0)(x - r_1)...(x - r_{n-1}) expressed
    in the Hermite_e basis.

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial, shape (n,).

    Returns
    -------
    HermitePolynomialHe
        Monic Hermite series with the given roots.

    Notes
    -----
    Uses NumPy's hermefromroots which constructs the polynomial from
    roots in the Hermite_e basis.

    Examples
    --------
    >>> roots = torch.tensor([0.5, -0.5])
    >>> c = hermite_polynomial_he_from_roots(roots)
    """
    # Use NumPy's hermefromroots
    roots_np = roots.detach().cpu().numpy()
    coeffs_np = np.polynomial.hermite_e.hermefromroots(roots_np)

    coeffs = torch.from_numpy(coeffs_np).to(
        dtype=roots.dtype if roots.is_floating_point() else torch.float32,
        device=roots.device,
    )

    return HermitePolynomialHe(coeffs=coeffs)

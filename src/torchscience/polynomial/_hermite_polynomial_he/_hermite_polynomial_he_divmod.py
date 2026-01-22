"""Division of Probabilists' Hermite series with remainder."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from ._hermite_polynomial_he import (
    HermitePolynomialHe,
    hermite_polynomial_he,
)


def hermite_polynomial_he_divmod(
    a: HermitePolynomialHe,
    b: HermitePolynomialHe,
) -> Tuple[HermitePolynomialHe, HermitePolynomialHe]:
    """Divide two Probabilists' Hermite series with remainder.

    Returns quotient q and remainder r such that a = b*q + r.

    Parameters
    ----------
    a : HermitePolynomialHe
        Dividend.
    b : HermitePolynomialHe
        Divisor.

    Returns
    -------
    Tuple[HermitePolynomialHe, HermitePolynomialHe]
        (quotient, remainder)

    Notes
    -----
    Uses NumPy's hermediv which performs polynomial division in the
    Hermite_e basis.

    Examples
    --------
    >>> a = hermite_polynomial_he(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = hermite_polynomial_he(torch.tensor([1.0, 1.0]))
    >>> q, r = hermite_polynomial_he_divmod(a, b)
    """
    # Convert to plain tensors for operations
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    # Use NumPy's hermediv
    a_np = a_coeffs.detach().cpu().numpy()
    b_np = b_coeffs.detach().cpu().numpy()

    q_np, r_np = np.polynomial.hermite_e.hermediv(a_np, b_np)

    q_coeffs = torch.from_numpy(q_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )
    r_coeffs = torch.from_numpy(r_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )

    return (
        hermite_polynomial_he(q_coeffs),
        hermite_polynomial_he(r_coeffs),
    )

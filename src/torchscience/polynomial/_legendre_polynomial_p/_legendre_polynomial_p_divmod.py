from typing import Tuple

import numpy as np
import torch

from ._legendre_polynomial_p import (
    LegendrePolynomialP,
    legendre_polynomial_p,
)


def legendre_polynomial_p_divmod(
    a: LegendrePolynomialP,
    b: LegendrePolynomialP,
) -> Tuple[LegendrePolynomialP, LegendrePolynomialP]:
    """Divide two Legendre series with remainder.

    Returns quotient q and remainder r such that a = b*q + r.

    Parameters
    ----------
    a : LegendrePolynomialP
        Dividend.
    b : LegendrePolynomialP
        Divisor.

    Returns
    -------
    Tuple[LegendrePolynomialP, LegendrePolynomialP]
        (quotient, remainder)

    Notes
    -----
    Uses NumPy's legdiv which performs polynomial division in the
    Legendre basis.

    Examples
    --------
    >>> a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = legendre_polynomial_p(torch.tensor([1.0, 1.0]))
    >>> q, r = legendre_polynomial_p_divmod(a, b)
    """
    # Convert to plain tensors to avoid operator interception
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    # Use NumPy's legdiv
    a_np = a_coeffs.detach().cpu().numpy()
    b_np = b_coeffs.detach().cpu().numpy()

    q_np, r_np = np.polynomial.legendre.legdiv(a_np, b_np)

    q_coeffs = torch.from_numpy(q_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )
    r_coeffs = torch.from_numpy(r_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )

    return (
        legendre_polynomial_p(q_coeffs),
        legendre_polynomial_p(r_coeffs),
    )

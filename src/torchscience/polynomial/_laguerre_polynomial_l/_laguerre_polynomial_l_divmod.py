from typing import Tuple

import numpy as np
import torch

from ._laguerre_polynomial_l import (
    LaguerrePolynomialL,
    laguerre_polynomial_l,
)


def laguerre_polynomial_l_divmod(
    a: LaguerrePolynomialL,
    b: LaguerrePolynomialL,
) -> Tuple[LaguerrePolynomialL, LaguerrePolynomialL]:
    """Divide two Laguerre series with remainder.

    Returns quotient q and remainder r such that a = b*q + r.

    Parameters
    ----------
    a : LaguerrePolynomialL
        Dividend.
    b : LaguerrePolynomialL
        Divisor.

    Returns
    -------
    Tuple[LaguerrePolynomialL, LaguerrePolynomialL]
        (quotient, remainder)

    Notes
    -----
    Uses NumPy's lagdiv which performs polynomial division in the
    Laguerre basis.

    Examples
    --------
    >>> a = laguerre_polynomial_l(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = laguerre_polynomial_l(torch.tensor([1.0, 1.0]))
    >>> q, r = laguerre_polynomial_l_divmod(a, b)
    """
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    # Use NumPy's lagdiv
    a_np = a_coeffs.detach().cpu().numpy()
    b_np = b_coeffs.detach().cpu().numpy()

    q_np, r_np = np.polynomial.laguerre.lagdiv(a_np, b_np)

    q_coeffs = torch.from_numpy(q_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )
    r_coeffs = torch.from_numpy(r_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )

    return (
        laguerre_polynomial_l(q_coeffs),
        laguerre_polynomial_l(r_coeffs),
    )

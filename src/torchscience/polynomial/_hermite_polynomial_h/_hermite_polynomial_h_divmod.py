from typing import Tuple

import numpy as np
import torch

from ._hermite_polynomial_h import (
    HermitePolynomialH,
    hermite_polynomial_h,
)


def hermite_polynomial_h_divmod(
    a: HermitePolynomialH,
    b: HermitePolynomialH,
) -> Tuple[HermitePolynomialH, HermitePolynomialH]:
    """Divide two Physicists' Hermite series with remainder.

    Returns quotient q and remainder r such that a = b*q + r.

    Parameters
    ----------
    a : HermitePolynomialH
        Dividend.
    b : HermitePolynomialH
        Divisor.

    Returns
    -------
    Tuple[HermitePolynomialH, HermitePolynomialH]
        (quotient, remainder)

    Notes
    -----
    Uses NumPy's hermdiv which performs polynomial division in the
    Hermite basis.

    Examples
    --------
    >>> a = hermite_polynomial_h(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = hermite_polynomial_h(torch.tensor([1.0, 1.0]))
    >>> q, r = hermite_polynomial_h_divmod(a, b)
    """
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    # Use NumPy's hermdiv
    a_np = a_coeffs.detach().cpu().numpy()
    b_np = b_coeffs.detach().cpu().numpy()

    q_np, r_np = np.polynomial.hermite.hermdiv(a_np, b_np)

    q_coeffs = torch.from_numpy(q_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )
    r_coeffs = torch.from_numpy(r_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )

    return (
        hermite_polynomial_h(q_coeffs),
        hermite_polynomial_h(r_coeffs),
    )

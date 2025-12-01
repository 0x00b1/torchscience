"""Multiply two Probabilists' Hermite series using linearization."""

from __future__ import annotations

import numpy as np
import torch

from ._hermite_polynomial_he import HermitePolynomialHe


def hermite_polynomial_he_multiply(
    a: HermitePolynomialHe,
    b: HermitePolynomialHe,
) -> HermitePolynomialHe:
    """Multiply two Probabilists' Hermite series.

    Uses NumPy's Hermite_e linearization for the product.

    Parameters
    ----------
    a : HermitePolynomialHe
        First series with coefficients a_0, a_1, ..., a_m.
    b : HermitePolynomialHe
        Second series with coefficients b_0, b_1, ..., b_n.

    Returns
    -------
    HermitePolynomialHe
        Product series with degree at most m + n.

    Notes
    -----
    The product of two Hermite series of degrees m and n has degree m + n.
    The linearization identity for Hermite polynomials ensures the product
    remains in Hermite form:

        He_m(x) * He_n(x) = sum_k c_k He_k(x)

    where the coefficients c_k are determined by the linearization formula.

    Examples
    --------
    >>> a = hermite_polynomial_he(torch.tensor([0.0, 1.0]))  # He_1 = x
    >>> b = hermite_polynomial_he(torch.tensor([0.0, 1.0]))  # He_1 = x
    >>> c = hermite_polynomial_he_multiply(a, b)
    >>> # He_1 * He_1 = x^2 = He_2 + 1 = He_2 + He_0
    """
    a_coeffs = a.coeffs
    b_coeffs = b.coeffs

    # Use NumPy's hermemul which implements the linearization formula
    # Note: numpy.polynomial.hermite_e uses the "probabilists'" convention
    a_np = a_coeffs.detach().cpu().numpy()
    b_np = b_coeffs.detach().cpu().numpy()

    result_np = np.polynomial.hermite_e.hermemul(a_np, b_np)

    result_coeffs = torch.from_numpy(result_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )

    return HermitePolynomialHe(coeffs=result_coeffs)

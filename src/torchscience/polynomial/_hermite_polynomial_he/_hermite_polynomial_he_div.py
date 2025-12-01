"""Quotient of Probabilists' Hermite series division."""

from __future__ import annotations

from ._hermite_polynomial_he import HermitePolynomialHe
from ._hermite_polynomial_he_divmod import hermite_polynomial_he_divmod


def hermite_polynomial_he_div(
    a: HermitePolynomialHe,
    b: HermitePolynomialHe,
) -> HermitePolynomialHe:
    """Divide two Probabilists' Hermite series, returning quotient only.

    Parameters
    ----------
    a : HermitePolynomialHe
        Dividend.
    b : HermitePolynomialHe
        Divisor.

    Returns
    -------
    HermitePolynomialHe
        Quotient.
    """
    q, _ = hermite_polynomial_he_divmod(a, b)
    return q

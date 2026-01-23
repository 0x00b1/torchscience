"""Divmod of Bernoulli polynomial series."""

from typing import Tuple

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
)
from ._bernoulli_polynomial_b_to_polynomial import (
    bernoulli_polynomial_b_to_polynomial,
)
from ._polynomial_to_bernoulli_polynomial_b import (
    polynomial_to_bernoulli_polynomial_b,
)


def bernoulli_polynomial_b_divmod(
    a: BernoulliPolynomialB,
    b: BernoulliPolynomialB,
) -> Tuple[BernoulliPolynomialB, BernoulliPolynomialB]:
    """Divmod of Bernoulli polynomial series.

    Computes quotient and remainder such that a = b * q + r.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Dividend.
    b : BernoulliPolynomialB
        Divisor.

    Returns
    -------
    Tuple[BernoulliPolynomialB, BernoulliPolynomialB]
        Quotient and remainder (q, r).
    """
    from torchscience.polynomial._polynomial._polynomial_divmod import (
        polynomial_divmod,
    )

    # Convert to standard polynomial basis
    p_a = bernoulli_polynomial_b_to_polynomial(a)
    p_b = bernoulli_polynomial_b_to_polynomial(b)

    # Perform divmod in standard basis
    p_q, p_r = polynomial_divmod(p_a, p_b)

    # Convert back to Bernoulli basis
    q = polynomial_to_bernoulli_polynomial_b(p_q)
    r = polynomial_to_bernoulli_polynomial_b(p_r)

    return q, r

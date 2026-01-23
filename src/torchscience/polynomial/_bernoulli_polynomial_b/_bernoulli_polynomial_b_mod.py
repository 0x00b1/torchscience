"""Modulo of Bernoulli polynomial series."""

from ._bernoulli_polynomial_b import BernoulliPolynomialB
from ._bernoulli_polynomial_b_divmod import bernoulli_polynomial_b_divmod


def bernoulli_polynomial_b_mod(
    a: BernoulliPolynomialB,
    b: BernoulliPolynomialB,
) -> BernoulliPolynomialB:
    """Modulo of Bernoulli polynomial series.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Dividend.
    b : BernoulliPolynomialB
        Divisor.

    Returns
    -------
    BernoulliPolynomialB
        Remainder a % b.
    """
    _, r = bernoulli_polynomial_b_divmod(a, b)
    return r

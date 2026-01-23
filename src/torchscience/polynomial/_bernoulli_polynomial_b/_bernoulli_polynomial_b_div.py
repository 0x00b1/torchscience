"""Integer division of Bernoulli polynomial series."""

from ._bernoulli_polynomial_b import BernoulliPolynomialB
from ._bernoulli_polynomial_b_divmod import bernoulli_polynomial_b_divmod


def bernoulli_polynomial_b_div(
    a: BernoulliPolynomialB,
    b: BernoulliPolynomialB,
) -> BernoulliPolynomialB:
    """Integer division of Bernoulli polynomial series.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Dividend.
    b : BernoulliPolynomialB
        Divisor.

    Returns
    -------
    BernoulliPolynomialB
        Quotient a // b.
    """
    q, _ = bernoulli_polynomial_b_divmod(a, b)
    return q

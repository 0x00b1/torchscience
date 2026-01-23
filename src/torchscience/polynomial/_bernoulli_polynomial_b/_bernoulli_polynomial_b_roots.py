"""Find roots of Bernoulli polynomial series."""

from torch import Tensor

from ._bernoulli_polynomial_b import BernoulliPolynomialB
from ._bernoulli_polynomial_b_to_polynomial import (
    bernoulli_polynomial_b_to_polynomial,
)


def bernoulli_polynomial_b_roots(a: BernoulliPolynomialB) -> Tensor:
    """Find roots of a Bernoulli polynomial series.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Series to find roots of.

    Returns
    -------
    Tensor
        Complex roots of the polynomial.

    Notes
    -----
    Converts to standard polynomial and finds roots via companion matrix.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([0.0, 1.0]))  # B_1(x) = x - 1/2
    >>> roots = bernoulli_polynomial_b_roots(a)
    >>> # Root should be x = 0.5
    """
    from torchscience.polynomial._polynomial._polynomial_roots import (
        polynomial_roots,
    )

    # Convert to standard polynomial
    p = bernoulli_polynomial_b_to_polynomial(a)

    # Find roots of standard polynomial
    return polynomial_roots(p)

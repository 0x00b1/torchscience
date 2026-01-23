"""Create Bernoulli polynomial series from roots."""

from torch import Tensor

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
)
from ._polynomial_to_bernoulli_polynomial_b import (
    polynomial_to_bernoulli_polynomial_b,
)


def bernoulli_polynomial_b_from_roots(roots: Tensor) -> BernoulliPolynomialB:
    """Create Bernoulli polynomial series from its roots.

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial.

    Returns
    -------
    BernoulliPolynomialB
        Bernoulli polynomial series with the given roots.

    Examples
    --------
    >>> roots = torch.tensor([0.5])  # B_1(x) = x - 0.5 has root at 0.5
    >>> a = bernoulli_polynomial_b_from_roots(roots)
    """
    from torchscience.polynomial._polynomial._polynomial_from_roots import (
        polynomial_from_roots,
    )

    # Create standard polynomial from roots
    p = polynomial_from_roots(roots)

    # Convert to Bernoulli basis
    return polynomial_to_bernoulli_polynomial_b(p)

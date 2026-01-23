"""Definite integral of Bernoulli polynomial series."""

from torch import Tensor

from ._bernoulli_polynomial_b import BernoulliPolynomialB
from ._bernoulli_polynomial_b_antiderivative import (
    bernoulli_polynomial_b_antiderivative,
)
from ._bernoulli_polynomial_b_evaluate import bernoulli_polynomial_b_evaluate


def bernoulli_polynomial_b_integral(
    a: BernoulliPolynomialB,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Compute definite integral of Bernoulli polynomial series.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Series to integrate.
    lower : Tensor
        Lower bound of integration.
    upper : Tensor
        Upper bound of integration.

    Returns
    -------
    Tensor
        Definite integral values.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([1.0]))  # B_0(x) = 1
    >>> bernoulli_polynomial_b_integral(a, torch.tensor(0.0), torch.tensor(1.0))
    tensor(1.)  # integral of 1 from 0 to 1
    """
    # Get antiderivative with constant 0
    A = bernoulli_polynomial_b_antiderivative(a, order=1, constant=0.0)

    # Evaluate at bounds
    upper_val = bernoulli_polynomial_b_evaluate(A, upper)
    lower_val = bernoulli_polynomial_b_evaluate(A, lower)

    return upper_val - lower_val

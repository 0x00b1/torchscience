"""Least squares fit of Bernoulli polynomial series."""

import torch
from torch import Tensor

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
    bernoulli_polynomial_b,
)
from ._bernoulli_polynomial_b_vandermonde import (
    bernoulli_polynomial_b_vandermonde,
)


def bernoulli_polynomial_b_fit(
    x: Tensor,
    y: Tensor,
    degree: int,
) -> BernoulliPolynomialB:
    """Least squares fit of Bernoulli polynomial series to data.

    Parameters
    ----------
    x : Tensor
        Sample points.
    y : Tensor
        Sample values at x.
    degree : int
        Degree of fitting polynomial.

    Returns
    -------
    BernoulliPolynomialB
        Fitted Bernoulli polynomial series.

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> y = torch.tensor([0.0, 0.25, 1.0])  # y = x^2
    >>> fit = bernoulli_polynomial_b_fit(x, y, 2)
    """
    # Build Vandermonde matrix
    V = bernoulli_polynomial_b_vandermonde(x, degree)

    # Solve least squares: V @ c = y
    # Using pseudo-inverse for numerical stability
    coeffs, _, _, _ = torch.linalg.lstsq(V, y.unsqueeze(-1))

    return bernoulli_polynomial_b(coeffs.squeeze(-1))

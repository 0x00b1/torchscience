from typing import Optional

import torch
from torch import Tensor

from ._polynomial import Polynomial, polynomial
from ._polynomial_vandermonde import polynomial_vandermonde


def polynomial_fit(
    x: Tensor,
    y: Tensor,
    degree: int,
    weights: Optional[Tensor] = None,
) -> Polynomial:
    """Fit polynomial to data using least squares.

    Finds polynomial p of given degree that minimizes
    sum_i weights[i] * (p(x[i]) - y[i])^2.

    Parameters
    ----------
    x : Tensor
        Sample x-coordinates, shape (n_points,).
    y : Tensor
        Sample y-values, shape (n_points,) or (n_points, *value_shape).
    degree : int
        Maximum polynomial degree.
    weights : Tensor, optional
        Weights for each sample, shape (n_points,).
        Default is uniform weights.

    Returns
    -------
    Polynomial
        Fitted polynomial with coefficients shape (degree + 1,) or
        (degree + 1, *value_shape) for vector-valued y.

    Raises
    ------
    ValueError
        If degree >= n_points (underdetermined system).

    Examples
    --------
    >>> x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> y = torch.tensor([1.0, 3.0, 5.0, 7.0])  # y = 1 + 2x
    >>> p = polynomial_fit(x, y, degree=1)
    >>> p
    Polynomial(tensor([1., 2.]))
    """
    n_points = x.shape[0]

    if degree >= n_points:
        raise ValueError(
            f"Degree {degree} must be less than number of points {n_points}"
        )

    # Build Vandermonde matrix
    V = polynomial_vandermonde(x, degree)  # (n_points, degree+1)

    # Handle y shape
    y_2d = y if y.dim() > 1 else y.unsqueeze(-1)

    # Apply weights if provided
    if weights is not None:
        sqrt_w = torch.sqrt(weights)
        V = V * sqrt_w.unsqueeze(-1)
        y_2d = y_2d * sqrt_w.unsqueeze(-1)

    # Solve least squares: V @ coeffs = y
    # Using torch.linalg.lstsq
    result = torch.linalg.lstsq(V, y_2d)
    coeffs = result.solution

    # Remove extra dimension if y was 1D
    if y.dim() == 1:
        coeffs = coeffs.squeeze(-1)

    return polynomial(coeffs)

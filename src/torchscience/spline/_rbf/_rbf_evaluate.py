"""RBF interpolator evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._rbf_kernels import CONDITIONALLY_POSITIVE_DEFINITE, evaluate_kernel

if TYPE_CHECKING:
    from ._rbf import RBFInterpolator


def rbf_evaluate(
    rbf: RBFInterpolator,
    query: Tensor,
) -> Tensor:
    """
    Evaluate an RBF interpolator at query points.

    Parameters
    ----------
    rbf : RBFInterpolator
        The RBF interpolator.
    query : Tensor
        Query points, shape (m, dim) or (m,) for 1D.

    Returns
    -------
    result : Tensor
        Interpolated values, shape (m, *value_shape).
    """
    centers = rbf.centers  # (n, dim)
    weights = rbf.weights  # (n, *value_shape)
    polynomial_coeffs = rbf.polynomial_coeffs
    kernel = rbf.kernel
    epsilon = rbf.epsilon

    n, dim = centers.shape

    # Handle 1D query
    if query.dim() == 1 and dim == 1:
        query = query.unsqueeze(-1)

    m = query.shape[0]

    # Get value shape
    if weights.dim() == 1:
        value_shape = ()
        weights_2d = weights.unsqueeze(-1)  # (n, 1)
        poly_2d = (
            polynomial_coeffs.unsqueeze(-1)
            if polynomial_coeffs.numel() > 0
            else polynomial_coeffs
        )
    else:
        value_shape = weights.shape[1:]
        weights_2d = weights.reshape(n, -1)  # (n, num_values)
        if polynomial_coeffs.numel() > 0:
            poly_2d = polynomial_coeffs.reshape(polynomial_coeffs.shape[0], -1)
        else:
            poly_2d = polynomial_coeffs

    num_values = weights_2d.shape[1]

    # Compute distances from query points to centers
    diff = query.unsqueeze(1) - centers.unsqueeze(0)  # (m, n, dim)
    distances = torch.norm(diff, dim=-1)  # (m, n)

    # Evaluate kernel
    K = evaluate_kernel(
        distances, kernel, epsilon if epsilon > 0 else None
    )  # (m, n)

    # Compute RBF contribution
    result = K @ weights_2d  # (m, num_values)

    # Add polynomial contribution for CPD kernels
    if (
        kernel in CONDITIONALLY_POSITIVE_DEFINITE
        and polynomial_coeffs.numel() > 0
    ):
        P = _build_polynomial_matrix(query, dim)  # (m, 1+dim)
        result = result + P @ poly_2d

    # Reshape result
    if value_shape:
        result = result.reshape(m, *value_shape)
    else:
        result = result.squeeze(-1)

    return result


def _build_polynomial_matrix(points: Tensor, dim: int) -> Tensor:
    """Build polynomial basis matrix for evaluation.

    Parameters
    ----------
    points : Tensor
        Query points, shape (m, dim).
    dim : int
        Spatial dimension.

    Returns
    -------
    P : Tensor
        Polynomial matrix, shape (m, 1+dim).
    """
    m = points.shape[0]
    P = torch.zeros(m, 1 + dim, dtype=points.dtype, device=points.device)
    P[:, 0] = 1.0
    P[:, 1:] = points
    return P

"""RBF interpolator fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from ._rbf_kernels import (
    CONDITIONALLY_POSITIVE_DEFINITE,
    KERNELS_WITH_EPSILON,
    evaluate_kernel,
)

if TYPE_CHECKING:
    from ._rbf import RBFInterpolator


def rbf_fit(
    points: Tensor,
    values: Tensor,
    kernel: str = "thin_plate",
    epsilon: Optional[float] = None,
    smoothing: float = 0.0,
) -> RBFInterpolator:
    """
    Fit an RBF interpolator to scattered data.

    Parameters
    ----------
    points : Tensor
        Data point locations, shape (n, dim).
    values : Tensor
        Data values, shape (n,) or (n, *value_shape).
    kernel : str
        RBF kernel name.
    epsilon : float, optional
        Shape parameter. If None and required, estimated from data.
    smoothing : float
        Smoothing parameter (regularization).

    Returns
    -------
    rbf : RBFInterpolator
        Fitted RBF interpolator.

    Notes
    -----
    For conditionally positive definite kernels (thin_plate, cubic, linear),
    polynomial terms are added to ensure well-posedness.

    The system solved is:
    [K + λI    P] [w]   [f]
    [P^T       0] [c] = [0]

    where K is the kernel matrix, P is the polynomial basis matrix,
    w are the RBF weights, c are polynomial coefficients, and λ is
    the smoothing parameter.
    """
    from ._rbf import RBFInterpolator

    n, dim = points.shape

    if n < 1:
        raise ValueError("Need at least 1 data point")

    # Handle value shape
    if values.dim() == 1:
        value_shape = ()
        values_2d = values.unsqueeze(-1)  # (n, 1)
    else:
        value_shape = values.shape[1:]
        values_2d = values.reshape(n, -1)  # (n, num_values)

    num_values = values_2d.shape[1]

    # Auto-select epsilon if needed
    if kernel in KERNELS_WITH_EPSILON and epsilon is None:
        epsilon = _estimate_epsilon(points)

    # Compute pairwise distances
    diff = points.unsqueeze(0) - points.unsqueeze(1)  # (n, n, dim)
    distances = torch.norm(diff, dim=-1)  # (n, n)

    # Build kernel matrix
    K = evaluate_kernel(distances, kernel, epsilon)

    # Add smoothing (regularization)
    if smoothing > 0:
        K = K + smoothing * torch.eye(n, dtype=K.dtype, device=K.device)

    # For conditionally positive definite kernels, add polynomial terms
    if kernel in CONDITIONALLY_POSITIVE_DEFINITE:
        poly_order = _get_polynomial_order(kernel)
        P = _build_polynomial_matrix(points, poly_order)  # (n, m)
        m = P.shape[1]

        # Build augmented system
        # [K  P] [w]   [f]
        # [P' 0] [c] = [0]
        A = torch.zeros(n + m, n + m, dtype=K.dtype, device=K.device)
        A[:n, :n] = K
        A[:n, n:] = P
        A[n:, :n] = P.T

        rhs = torch.zeros(
            n + m, num_values, dtype=values_2d.dtype, device=values_2d.device
        )
        rhs[:n, :] = values_2d

        # Solve the system
        solution = torch.linalg.solve(A, rhs)

        weights = solution[:n]  # (n, num_values)
        polynomial_coeffs = solution[n:]  # (m, num_values)
    else:
        # Positive definite kernels: no polynomial needed
        weights = torch.linalg.solve(K, values_2d)
        polynomial_coeffs = torch.zeros(
            0, num_values, dtype=weights.dtype, device=weights.device
        )

    # Reshape weights and polynomial coefficients
    if value_shape:
        weights = weights.reshape(n, *value_shape)
        if polynomial_coeffs.shape[0] > 0:
            polynomial_coeffs = polynomial_coeffs.reshape(-1, *value_shape)
        else:
            polynomial_coeffs = polynomial_coeffs.reshape(0, *value_shape)
    else:
        weights = weights.squeeze(-1)
        polynomial_coeffs = polynomial_coeffs.squeeze(-1)

    return RBFInterpolator(
        centers=points.clone(),
        weights=weights,
        polynomial_coeffs=polynomial_coeffs,
        kernel=kernel,
        epsilon=epsilon if epsilon is not None else 0.0,
        batch_size=[],
    )


def _estimate_epsilon(points: Tensor) -> float:
    """Estimate epsilon from average nearest neighbor distance."""
    n = points.shape[0]
    if n < 2:
        return 1.0

    # Compute pairwise distances
    diff = points.unsqueeze(0) - points.unsqueeze(1)
    distances = torch.norm(diff, dim=-1)

    # Set diagonal to large value to exclude self-distances
    mask = torch.eye(n, dtype=torch.bool, device=distances.device)
    distances = distances.masked_fill(mask, float("inf"))

    # Average nearest neighbor distance
    min_distances, _ = distances.min(dim=1)

    # Filter out infinite values
    finite_mask = torch.isfinite(min_distances)
    if not finite_mask.any():
        return 1.0

    avg_dist = min_distances[finite_mask].mean().item()

    # Epsilon inversely related to typical distance
    # Ensure result is finite and positive
    if avg_dist <= 0 or not torch.isfinite(torch.tensor(avg_dist)):
        return 1.0

    return 1.0 / avg_dist


def _get_polynomial_order(kernel: str) -> int:
    """Get polynomial order for conditionally positive definite kernels."""
    if kernel == "linear":
        return 1  # Constant + linear terms
    elif kernel in ("thin_plate", "cubic"):
        return 1  # Constant + linear terms (order 1 polynomial)
    else:
        return 0


def _build_polynomial_matrix(points: Tensor, order: int) -> Tensor:
    """Build polynomial basis matrix.

    Parameters
    ----------
    points : Tensor
        Points, shape (n, dim).
    order : int
        Polynomial order.

    Returns
    -------
    P : Tensor
        Polynomial matrix, shape (n, m).
    """
    n, dim = points.shape

    if order == 0:
        # Constant only
        return torch.ones(n, 1, dtype=points.dtype, device=points.device)
    elif order == 1:
        # Constant + linear terms: [1, x1, x2, ..., xd]
        P = torch.zeros(n, 1 + dim, dtype=points.dtype, device=points.device)
        P[:, 0] = 1.0
        P[:, 1:] = points
        return P
    else:
        raise ValueError(f"Polynomial order {order} not implemented")

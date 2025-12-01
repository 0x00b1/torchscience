from typing import Optional

import torch
from torch import Tensor

from .._degree_error import DegreeError
from .._knot_error import KnotError


def b_spline_basis(
    t: Tensor,
    knots: Tensor,
    degree: int,
    i: Optional[int] = None,
) -> Tensor:
    """
    Evaluate B-spline basis functions using Cox-de Boor recursion.

    Parameters
    ----------
    t : Tensor
        Evaluation points, shape (*query_shape)
    knots : Tensor
        Knot vector, shape (n_knots,). Must be non-decreasing.
    degree : int
        Polynomial degree (0=constant, 1=linear, 2=quadratic, 3=cubic)
    i : int, optional
        If specified, return only the i-th basis function. Otherwise return all.

    Returns
    -------
    basis : Tensor
        If i is None: shape (*query_shape, n_basis) where n_basis = n_knots - degree - 1
        If i is specified: shape (*query_shape)

    Raises
    ------
    DegreeError
        If degree is negative or too high for the given knot count.
    KnotError
        If knots are not non-decreasing.

    Notes
    -----
    For degree 0:
        B_{i,0}(t) = 1 if t_i <= t < t_{i+1}, else 0

    For degree k > 0:
        B_{i,k}(t) = ((t - t_i) / (t_{i+k} - t_i)) * B_{i,k-1}(t)
                   + ((t_{i+k+1} - t) / (t_{i+k+1} - t_{i+1})) * B_{i+1,k-1}(t)

    Division by zero (0/0) is handled as 0 (when knot intervals are zero).

    The implementation uses dynamic programming (bottom-up evaluation) rather
    than naive recursion for efficiency.
    """
    n_knots = knots.shape[0]

    # Validate degree
    if degree < 0:
        raise DegreeError(f"Degree must be non-negative, got {degree}")
    if n_knots < degree + 2:
        raise DegreeError(
            f"Need at least {degree + 2} knots for degree {degree}, got {n_knots}"
        )

    # Validate knots are non-decreasing
    if not torch.all(knots[1:] >= knots[:-1]):
        raise KnotError("Knots must be non-decreasing")

    # Number of basis functions
    n_basis = n_knots - degree - 1

    # Handle scalar input
    is_scalar = t.dim() == 0
    if is_scalar:
        t = t.unsqueeze(0)

    # Store original query shape and flatten for computation
    query_shape = t.shape
    t_flat = t.flatten()  # (n_points,)
    n_points = t_flat.shape[0]

    # Initialize degree 0 basis functions
    # B_{j,0}(t) = 1 if t_j <= t < t_{j+1}, else 0
    # We need (n_knots - 1) degree-0 basis functions for the recursion
    # Shape: (n_points, n_knots - 1)

    # Use >= for left boundary and < for right boundary
    # Handle rightmost point special case: t == knots[-1] should be in the last interval
    t_expanded = t_flat.unsqueeze(-1)  # (n_points, 1)
    knots_left = knots[:-1].unsqueeze(0)  # (1, n_knots - 1)
    knots_right = knots[1:].unsqueeze(0)  # (1, n_knots - 1)

    # Standard interval check: t_i <= t < t_{i+1}
    basis_0 = ((t_expanded >= knots_left) & (t_expanded < knots_right)).to(
        dtype=knots.dtype
    )

    # Handle rightmost point: if t == knots[-1], include in last interval
    # Find where t equals the last knot
    at_right_boundary = t_flat == knots[-1]
    if at_right_boundary.any():
        # Find the last non-empty interval (where knots differ)
        # For each rightmost point, set the last interval with span > 0 to 1
        # But for simplicity, set the rightmost interval to 1
        # Actually, we need to find the rightmost interval that exists
        # For uniform knots, this is simply the last one
        # For repeated knots at the end, we need to find the last interval with positive span
        last_valid_interval = n_knots - 2
        for j in range(n_knots - 2, -1, -1):
            if knots[j] < knots[j + 1]:
                last_valid_interval = j
                break
        basis_0[at_right_boundary, last_valid_interval] = 1.0

    # Dynamic programming: build up from degree 0 to target degree
    # At each level k, we compute B_{j,k} for j = 0, 1, ..., n_knots - k - 2
    # The number of basis functions at degree k is: n_knots - k - 1

    basis_current = basis_0  # Shape: (n_points, n_knots - 1)

    for k in range(1, degree + 1):
        # Number of basis functions at this degree
        n_basis_k = n_knots - k - 1

        # Allocate new basis tensor
        basis_next = torch.zeros(
            n_points, n_basis_k, dtype=knots.dtype, device=knots.device
        )

        for j in range(n_basis_k):
            # Left term: ((t - t_j) / (t_{j+k} - t_j)) * B_{j,k-1}(t)
            denom_left = knots[j + k] - knots[j]
            if denom_left.abs() > 0:
                alpha_left = (t_flat - knots[j]) / denom_left
                left_term = alpha_left * basis_current[:, j]
            else:
                # 0/0 case: treat as 0
                left_term = torch.zeros(
                    n_points, dtype=knots.dtype, device=knots.device
                )

            # Right term: ((t_{j+k+1} - t) / (t_{j+k+1} - t_{j+1})) * B_{j+1,k-1}(t)
            denom_right = knots[j + k + 1] - knots[j + 1]
            if denom_right.abs() > 0:
                alpha_right = (knots[j + k + 1] - t_flat) / denom_right
                right_term = alpha_right * basis_current[:, j + 1]
            else:
                # 0/0 case: treat as 0
                right_term = torch.zeros(
                    n_points, dtype=knots.dtype, device=knots.device
                )

            basis_next[:, j] = left_term + right_term

        basis_current = basis_next

    # basis_current now has shape (n_points, n_basis)
    # Reshape back to (*query_shape, n_basis)
    result = basis_current.view(*query_shape, n_basis)

    # Handle scalar input: remove the leading dimension
    if is_scalar:
        result = result.squeeze(0)

    # If i is specified, return only the i-th basis function
    if i is not None:
        if i < 0 or i >= n_basis:
            raise IndexError(
                f"Basis index {i} out of range [0, {n_basis - 1}]"
            )
        # Select the i-th basis function
        result = result[..., i]

    return result

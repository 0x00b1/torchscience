"""Collocation coefficients and vectorized residual computation."""

from functools import lru_cache
from typing import Tuple

import torch
from torch import Tensor

# Lobatto IIIA 4th-order coefficients (3 stages)
# Nodes: c = [0, 1/2, 1]
# This is the same method used by scipy's solve_bvp
_C_RAW = [0.0, 0.5, 1.0]

# Runge-Kutta matrix A
# For Lobatto IIIA with 3 stages:
#   A[0,:] = [0, 0, 0]           (explicit first stage)
#   A[1,:] = [5/24, 1/3, -1/24]  (implicit middle stage)
#   A[2,:] = [1/6, 2/3, 1/6]     (implicit last stage = quadrature weights)
_A_RAW = [
    [0.0, 0.0, 0.0],
    [5 / 24, 1 / 3, -1 / 24],
    [1 / 6, 2 / 3, 1 / 6],
]

# Quadrature weights B (same as last row of A for Lobatto)
_B_RAW = [1 / 6, 2 / 3, 1 / 6]


@lru_cache(maxsize=8)
def get_lobatto_coefficients(
    dtype_str: str,
    device_type: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Get Lobatto IIIA collocation coefficients.

    Note: Uses string keys for proper LRU cache hashing, matching IVP's
    `_get_tableau()` pattern from `_dormand_prince_5.py`.

    Parameters
    ----------
    dtype_str : str
        Data type as string (e.g., "float32", "float64").
    device_type : str
        Device type ("cpu" or "cuda").

    Returns
    -------
    c : Tensor
        Collocation nodes, shape (3,). Values in [0, 1].
    A : Tensor
        Runge-Kutta matrix, shape (3, 3).
    B : Tensor
        Quadrature weights, shape (3,).
    """
    # Convert string keys back to torch types (matches IVP pattern)
    dtype = getattr(torch, dtype_str)
    device = torch.device(device_type)
    c = torch.tensor(_C_RAW, dtype=dtype, device=device)
    A = torch.tensor(_A_RAW, dtype=dtype, device=device)
    B = torch.tensor(_B_RAW, dtype=dtype, device=device)
    return c, A, B


def _get_coefficients(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Helper to get Lobatto coefficients for a tensor's dtype/device."""
    dtype_str = str(x.dtype).replace("torch.", "")
    device_str = str(x.device.type)
    return get_lobatto_coefficients(dtype_str, device_str)


def compute_collocation_residual(
    fun: "Callable[[Tensor, Tensor, Tensor], Tensor]",
    x: Tensor,
    y: Tensor,
    p: Tensor,
) -> Tensor:
    """Compute collocation residual for Lobatto IIIA method (vectorized).

    For 4th-order Lobatto IIIA, the residual at the midpoint of each interval
    measures how well the cubic Hermite interpolant satisfies the ODE at the
    collocation point.

    Parameters
    ----------
    fun : callable
        RHS of ODE: dy/dx = fun(x, y, p). Must be vectorized.
    x : Tensor
        Mesh nodes, shape (n_nodes,).
    y : Tensor
        Solution values, shape (n_components, n_nodes).
    p : Tensor
        Parameters, shape (n_params,) or empty.

    Returns
    -------
    Tensor
        Collocation residual, shape (n_components, n_intervals).

    Notes
    -----
    For 3-point Lobatto IIIA (4th order), the collocation equation at the
    midpoint is:

        y_mid = (y_left + y_right)/2 + h/8 * (f_left - f_right)

    The residual is the difference between the Hermite-interpolated midpoint
    value and this expression. When the residual is zero, the cubic Hermite
    interpolant exactly satisfies the ODE at the midpoint.

    This is fully vectorized - fun is called once with all mesh points.
    """
    n_nodes = x.shape[0]
    n_intervals = n_nodes - 1

    # Compute h for each interval
    h = x[1:] - x[:-1]  # (n_intervals,)

    # Evaluate f at all mesh nodes (single vectorized call)
    f_nodes = fun(x, y, p)  # (n_components, n_nodes)

    # Extract left and right values for each interval
    y_left = y[:, :-1]  # (n_components, n_intervals)
    y_right = y[:, 1:]  # (n_components, n_intervals)
    f_left = f_nodes[:, :-1]  # (n_components, n_intervals)
    f_right = f_nodes[:, 1:]  # (n_components, n_intervals)

    # Compute midpoint using Hermite interpolation
    # For cubic Hermite at t=0.5:
    #   H00(0.5) = (1 + 2*0.5)(0.5)^2 = 0.5
    #   H10(0.5) = 0.5 * (0.5)^2 = 0.125
    #   H01(0.5) = (0.5)^2 * (3 - 2*0.5) = 0.5
    #   H11(0.5) = (0.5)^2 * (0.5 - 1) = -0.125
    # y_mid = 0.5*y_left + 0.125*h*f_left + 0.5*y_right - 0.125*h*f_right
    h_expanded = h.unsqueeze(0)  # (1, n_intervals)
    y_mid_hermite = (
        0.5 * y_left
        + 0.125 * h_expanded * f_left
        + 0.5 * y_right
        - 0.125 * h_expanded * f_right
    )

    # Collocation condition for 4th-order Lobatto:
    # y_mid should satisfy: y_mid = (y_left + y_right)/2 + h/8 * (f_left - f_right)
    # This is what the Hermite interpolant gives, so we need to check if
    # the derivative at mid matches the ODE.

    # Compute midpoint x values
    x_mid = (x[:-1] + x[1:]) / 2  # (n_intervals,)

    # Evaluate f at midpoint
    f_mid = fun(x_mid, y_mid_hermite, p)  # (n_components, n_intervals)

    # The collocation residual is the difference between:
    # 1. The RHS evaluated at midpoint: f_mid
    # 2. The derivative of cubic Hermite at midpoint
    #
    # For cubic Hermite derivative at t=0.5:
    #   dy/dt at t=0.5, then divide by h to get dy/dx
    #   H00'(0.5) = 6*(0.5)*(0.5-1) = -1.5
    #   H10'(0.5) = (1-0.5)^2 - 2*0.5*(1-0.5) = 0.25 - 0.5 = -0.25
    #   H01'(0.5) = 6*0.5*(1-0.5) = 1.5
    #   H11'(0.5) = 2*0.5*(0.5-1) + (0.5)^2 = -0.5 + 0.25 = -0.25
    #
    # dy/dt_mid = -1.5*y_left - 0.25*h*f_left + 1.5*y_right - 0.25*h*f_right
    # dy/dx_mid = (1/h) * dy/dt_mid

    dydt_mid = (
        -1.5 * y_left
        - 0.25 * h_expanded * f_left
        + 1.5 * y_right
        - 0.25 * h_expanded * f_right
    )
    dydx_mid = dydt_mid / h_expanded

    # Residual: how much the Hermite derivative differs from the ODE RHS
    residual = dydx_mid - f_mid

    return residual


def compute_collocation_residual_and_f(
    fun: "Callable[[Tensor, Tensor, Tensor], Tensor]",
    x: Tensor,
    y: Tensor,
    p: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute collocation residual and f values at left/mid/right.

    This is used by mesh refinement which needs f values for interpolation.

    Returns
    -------
    residual : Tensor
        Collocation residual, shape (n_components, n_intervals).
    f_left : Tensor
        f values at left nodes, shape (n_components, n_intervals).
    f_mid : Tensor
        f values at midpoints, shape (n_components, n_intervals).
    f_right : Tensor
        f values at right nodes, shape (n_components, n_intervals).
    """
    n_nodes = x.shape[0]

    # Compute h for each interval
    h = x[1:] - x[:-1]  # (n_intervals,)

    # Evaluate f at all mesh nodes
    f_nodes = fun(x, y, p)  # (n_components, n_nodes)

    # Extract left and right values
    y_left = y[:, :-1]
    y_right = y[:, 1:]
    f_left = f_nodes[:, :-1]
    f_right = f_nodes[:, 1:]

    # Compute midpoint using Hermite interpolation
    h_expanded = h.unsqueeze(0)
    y_mid_hermite = (
        0.5 * y_left
        + 0.125 * h_expanded * f_left
        + 0.5 * y_right
        - 0.125 * h_expanded * f_right
    )

    # Compute midpoint x values
    x_mid = (x[:-1] + x[1:]) / 2

    # Evaluate f at midpoint
    f_mid = fun(x_mid, y_mid_hermite, p)

    # Compute Hermite derivative at midpoint
    dydt_mid = (
        -1.5 * y_left
        - 0.25 * h_expanded * f_left
        + 1.5 * y_right
        - 0.25 * h_expanded * f_right
    )
    dydx_mid = dydt_mid / h_expanded

    # Residual
    residual = dydx_mid - f_mid

    return residual, f_left, f_mid, f_right

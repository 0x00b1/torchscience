"""Mesh adaptation for BVP solver."""

from typing import Tuple

import torch
from torch import Tensor

from torchscience.ordinary_differential_equation._interpolation import (
    hermite_interpolate,
)


def compute_rms_residuals(residual: Tensor) -> Tensor:
    """Compute RMS residual for each interval.

    Parameters
    ----------
    residual : Tensor
        Collocation residual, shape (n_components, n_intervals).

    Returns
    -------
    Tensor
        RMS residual per interval, shape (n_intervals,).
    """
    # RMS over components
    return torch.sqrt((residual**2).mean(dim=0))


def refine_mesh(
    x: Tensor,
    y: Tensor,
    f: Tensor,
    rms_residuals: Tensor,
    tol: float,
) -> Tuple[Tensor, Tensor]:
    """Refine mesh by inserting nodes in intervals with high residuals.

    Uses cubic Hermite interpolation to estimate y at new nodes,
    preserving 4th-order accuracy.

    Parameters
    ----------
    x : Tensor
        Current mesh nodes, shape (n_nodes,).
    y : Tensor
        Solution values, shape (n_components, n_nodes).
    f : Tensor
        Derivative values (dy/dx), shape (n_components, n_nodes).
    rms_residuals : Tensor
        RMS residuals per interval, shape (n_intervals,).
    tol : float
        Tolerance for refinement decision.

    Returns
    -------
    x_new : Tensor
        New mesh nodes, shape (n_new_nodes,).
    y_new : Tensor
        Interpolated y values, shape (n_components, n_new_nodes).
    """
    n_nodes = x.shape[0]
    n_intervals = n_nodes - 1

    # Find intervals that need refinement
    # Refine if residual > tol
    needs_refine = rms_residuals > tol

    if not needs_refine.any():
        # No refinement needed
        return x, y

    # Build new mesh by inserting midpoints where needed
    new_x_list = [x[0:1]]  # Start with first node
    new_y_list = [y[:, 0:1]]

    for i in range(n_intervals):
        if needs_refine[i]:
            # Insert midpoint
            x_mid = (x[i] + x[i + 1]) / 2

            # Interpolate y at midpoint using cubic Hermite
            x_query = x_mid.unsqueeze(0)
            y_mid = hermite_interpolate(x, y, f, x_query)

            new_x_list.append(x_mid.unsqueeze(0))
            new_y_list.append(y_mid)

        # Always add the right endpoint
        new_x_list.append(x[i + 1 : i + 2])
        new_y_list.append(y[:, i + 1 : i + 2])

    x_new = torch.cat(new_x_list)
    y_new = torch.cat(new_y_list, dim=1)

    return x_new, y_new

from typing import Union

import torch
from torch import Tensor

from ._cubic_spline import CubicSpline


def cubic_spline_integral(
    spline: CubicSpline,
    a: Union[float, Tensor],
    b: Union[float, Tensor],
) -> Tensor:
    """
    Compute the definite integral of a cubic spline from a to b.

    Parameters
    ----------
    spline : CubicSpline
        Input cubic spline
    a : float or Tensor
        Lower bound of integration
    b : float or Tensor
        Upper bound of integration

    Returns
    -------
    integral : Tensor
        Definite integral value(s), shape (*y_dim) or batched

    Notes
    -----
    For cubic polynomial on segment [x_i, x_{i+1}]:
        y = a + b*dx + c*dx^2 + d*dx^3

    The antiderivative is:
        F(dx) = a*dx + (b/2)*dx^2 + (c/3)*dx^3 + (d/4)*dx^4

    For definite integral:
        integral[t1, t2] = F(t2 - x_i) - F(t1 - x_i)

    If [a, b] spans multiple segments, the integrals are summed.
    """
    knots = spline.knots
    coeffs = spline.coefficients
    n_segments = len(knots) - 1

    # Convert bounds to tensors if necessary
    if not isinstance(a, Tensor):
        a = torch.tensor(a, dtype=knots.dtype, device=knots.device)
    if not isinstance(b, Tensor):
        b = torch.tensor(b, dtype=knots.dtype, device=knots.device)

    # Handle a > b case: integral from b to a = -integral from a to b
    sign = torch.ones(1, dtype=knots.dtype, device=knots.device)
    if a > b:
        a, b = b, a
        sign = -sign

    # Handle a == b case
    if a == b:
        # Return zero with correct shape
        if coeffs.dim() > 2:
            value_shape = coeffs.shape[2:]
            return torch.zeros(
                value_shape, dtype=knots.dtype, device=knots.device
            )
        else:
            return torch.tensor(0.0, dtype=knots.dtype, device=knots.device)

    # Clamp bounds to spline domain
    t_min = knots[0]
    t_max = knots[-1]
    a_clamped = torch.clamp(a, t_min, t_max)
    b_clamped = torch.clamp(b, t_min, t_max)

    # Find segment indices for lower and upper bounds
    # searchsorted returns index where value would be inserted
    seg_a = torch.searchsorted(knots, a_clamped, right=True) - 1
    seg_b = torch.searchsorted(knots, b_clamped, right=True) - 1

    # Clamp to valid segment range
    seg_a = torch.clamp(seg_a, 0, n_segments - 1)
    seg_b = torch.clamp(seg_b, 0, n_segments - 1)

    # Get value shape for multi-dimensional output
    if coeffs.dim() > 2:
        value_shape = coeffs.shape[2:]
    else:
        value_shape = ()

    # Initialize total integral
    if value_shape:
        total = torch.zeros(
            value_shape, dtype=knots.dtype, device=knots.device
        )
    else:
        total = torch.tensor(0.0, dtype=knots.dtype, device=knots.device)

    def antiderivative(dx: Tensor, seg_idx: int) -> Tensor:
        """
        Evaluate the antiderivative F(dx) = a*dx + (b/2)*dx^2 + (c/3)*dx^3 + (d/4)*dx^4

        Parameters
        ----------
        dx : Tensor
            Offset from segment start (t - x_i)
        seg_idx : int
            Segment index

        Returns
        -------
        F : Tensor
            Antiderivative value at dx
        """
        c_a = coeffs[seg_idx, 0]  # constant term
        c_b = coeffs[seg_idx, 1]  # linear term
        c_c = coeffs[seg_idx, 2]  # quadratic term
        c_d = coeffs[seg_idx, 3]  # cubic term

        if value_shape:
            # Expand dx for broadcasting with value dimensions
            dx_expanded = dx.view(*([1] * len(value_shape)))
        else:
            dx_expanded = dx

        # F(dx) = a*dx + (b/2)*dx^2 + (c/3)*dx^3 + (d/4)*dx^4
        return (
            c_a * dx_expanded
            + (c_b / 2) * dx_expanded**2
            + (c_c / 3) * dx_expanded**3
            + (c_d / 4) * dx_expanded**4
        )

    # Integrate over each segment that intersects [a, b]
    for seg_idx in range(seg_a.item(), seg_b.item() + 1):
        # Determine integration bounds within this segment
        seg_start = knots[seg_idx]
        seg_end = knots[seg_idx + 1]

        # Lower bound for this segment: max(a, seg_start)
        lower = torch.max(a_clamped, seg_start)

        # Upper bound for this segment: min(b, seg_end)
        upper = torch.min(b_clamped, seg_end)

        # Compute offsets from segment start
        dx_lower = lower - seg_start
        dx_upper = upper - seg_start

        # Add contribution: F(upper - seg_start) - F(lower - seg_start)
        contribution = antiderivative(dx_upper, seg_idx) - antiderivative(
            dx_lower, seg_idx
        )
        total = total + contribution

    return sign.squeeze() * total

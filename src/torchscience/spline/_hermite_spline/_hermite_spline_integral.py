"""Hermite spline definite integral computation."""

from typing import Union

import torch
from torch import Tensor

from ._hermite_spline import HermiteSpline


def hermite_spline_integral(
    spline: HermiteSpline,
    a: Union[float, Tensor],
    b: Union[float, Tensor],
) -> Tensor:
    """
    Compute the definite integral of a Hermite spline from a to b.

    Parameters
    ----------
    spline : HermiteSpline
        Input Hermite spline
    a : float or Tensor
        Lower bound of integration
    b : float or Tensor
        Upper bound of integration

    Returns
    -------
    integral : Tensor
        Definite integral value(s)

    Notes
    -----
    For Hermite interpolation on [x_i, x_{i+1}] with h = x_{i+1} - x_i:
    p(x) = H_00(u)*y_i + H_10(u)*h*d_i + H_01(u)*y_{i+1} + H_11(u)*h*d_{i+1}

    where u = (x - x_i) / h.

    The antiderivative is (with respect to x):
    P(x) = h * [I_00(u)*y_i + I_10(u)*h*d_i + I_01(u)*y_{i+1} + I_11(u)*h*d_{i+1}]

    where I_jk are the antiderivatives of H_jk:
    I_00(u) = integral of (2u^3 - 3u^2 + 1) = u^4/2 - u^3 + u
    I_10(u) = integral of (u^3 - 2u^2 + u) = u^4/4 - 2u^3/3 + u^2/2
    I_01(u) = integral of (-2u^3 + 3u^2) = -u^4/2 + u^3
    I_11(u) = integral of (u^3 - u^2) = u^4/4 - u^3/3
    """
    knots = spline.knots
    y_vals = spline.y
    derivatives = spline.dydx
    n_segments = len(knots) - 1

    # Convert bounds to tensors if necessary
    if not isinstance(a, Tensor):
        a = torch.tensor(a, dtype=knots.dtype, device=knots.device)
    if not isinstance(b, Tensor):
        b = torch.tensor(b, dtype=knots.dtype, device=knots.device)

    # Handle a > b case
    sign = torch.ones(1, dtype=knots.dtype, device=knots.device)
    if a > b:
        a, b = b, a
        sign = -sign

    # Handle a == b case
    if a == b:
        if y_vals.dim() > 1:
            value_shape = y_vals.shape[1:]
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

    # Find segment indices
    seg_a = torch.searchsorted(knots, a_clamped, right=True) - 1
    seg_b = torch.searchsorted(knots, b_clamped, right=True) - 1

    seg_a = torch.clamp(seg_a, 0, n_segments - 1)
    seg_b = torch.clamp(seg_b, 0, n_segments - 1)

    # Get value shape
    if y_vals.dim() > 1:
        value_shape = y_vals.shape[1:]
    else:
        value_shape = ()

    # Initialize total integral
    if value_shape:
        total = torch.zeros(
            value_shape, dtype=knots.dtype, device=knots.device
        )
    else:
        total = torch.tensor(0.0, dtype=knots.dtype, device=knots.device)

    def antiderivative(u: Tensor, seg_idx: int) -> Tensor:
        """
        Evaluate the antiderivative at normalized parameter u for segment seg_idx.

        Returns h * [I_00(u)*y_i + I_10(u)*h*d_i + I_01(u)*y_{i+1} + I_11(u)*h*d_{i+1}]
        """
        y_i = y_vals[seg_idx]
        y_ip1 = y_vals[seg_idx + 1]
        d_i = derivatives[seg_idx]
        d_ip1 = derivatives[seg_idx + 1]
        h = knots[seg_idx + 1] - knots[seg_idx]

        if value_shape:
            u_exp = u.view(*([1] * len(value_shape)))
            h_exp = h.view(*([1] * len(value_shape)))
        else:
            u_exp = u
            h_exp = h

        u2 = u_exp * u_exp
        u3 = u2 * u_exp
        u4 = u3 * u_exp

        # Antiderivatives of Hermite basis functions
        i_00 = u4 / 2 - u3 + u_exp
        i_10 = u4 / 4 - 2 * u3 / 3 + u2 / 2
        i_01 = -u4 / 2 + u3
        i_11 = u4 / 4 - u3 / 3

        return h_exp * (
            i_00 * y_i
            + i_10 * h_exp * d_i
            + i_01 * y_ip1
            + i_11 * h_exp * d_ip1
        )

    # Integrate over each segment that intersects [a, b]
    for seg_idx in range(seg_a.item(), seg_b.item() + 1):
        seg_start = knots[seg_idx]
        seg_end = knots[seg_idx + 1]
        h = seg_end - seg_start

        lower = torch.max(a_clamped, seg_start)
        upper = torch.min(b_clamped, seg_end)

        # Convert to normalized parameters
        u_lower = (lower - seg_start) / h
        u_upper = (upper - seg_start) / h

        contribution = antiderivative(u_upper, seg_idx) - antiderivative(
            u_lower, seg_idx
        )
        total = total + contribution

    return sign.squeeze() * total

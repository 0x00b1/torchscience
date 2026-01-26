"""Line search utilities for optimization solvers."""

from typing import Callable, Tuple

import torch
from torch import Tensor


def _cubic_interpolate(
    x1: Tensor,
    f1: Tensor,
    g1: Tensor,
    x2: Tensor,
    f2: Tensor,
    g2: Tensor,
) -> Tensor:
    """Cubic interpolation between two points with function values and derivatives.

    Finds the minimizer of the cubic polynomial interpolating ``(x1, f1, g1)``
    and ``(x2, f2, g2)``. Falls back to bisection if the cubic has no real
    minimizer.

    Parameters
    ----------
    x1, x2 : Tensor
        Step lengths.
    f1, f2 : Tensor
        Function values at ``x1`` and ``x2``.
    g1, g2 : Tensor
        Directional derivatives at ``x1`` and ``x2``.

    Returns
    -------
    Tensor
        The step length that minimizes the cubic interpolant.
    """
    d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2)
    d2_sq = d1 * d1 - g1 * g2
    # Clamp to avoid sqrt of negative (fallback to bisection)
    d2_sq = torch.clamp(d2_sq, min=0.0)
    d2 = torch.sqrt(d2_sq)
    # Minimizer of the cubic
    min_pos = x2 - (x2 - x1) * (g2 + d2 - d1) / (g2 - g1 + 2.0 * d2)
    return min_pos


def _backtracking_line_search(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    direction: Tensor,
    grad_dot_dir: Tensor,
    f_x: Tensor,
    *,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_backtracks: int = 20,
) -> Tensor:
    """Backtracking line search satisfying the Armijo condition.

    Finds a step length ``alpha`` such that:

    .. math::

        f(x + \\alpha d) \\leq f(x) + c_1 \\alpha \\nabla f(x)^T d

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Objective function.
    x : Tensor
        Current iterate of shape ``(..., n)``.
    direction : Tensor
        Search direction of shape ``(..., n)``.
    grad_dot_dir : Tensor
        Directional derivative ``grad^T @ direction`` of shape ``(...,)``.
    f_x : Tensor
        Function value at ``x`` of shape ``(...,)``.
    alpha_init : float
        Initial step length. Default: 1.0.
    c1 : float
        Armijo condition parameter. Default: 1e-4.
    rho : float
        Backtracking factor. Default: 0.5.
    max_backtracks : int
        Maximum number of backtracking steps. Default: 20.

    Returns
    -------
    Tensor
        Step length ``alpha`` of shape ``(...,)`` or scalar.
    """
    alpha = torch.full_like(f_x, alpha_init)
    done = torch.zeros_like(f_x, dtype=torch.bool)

    for _ in range(max_backtracks):
        # Evaluate at candidate point
        alpha_expanded = alpha.unsqueeze(-1) if x.dim() > 1 else alpha
        x_new = x + alpha_expanded * direction
        f_new = f(x_new)

        # Armijo sufficient decrease condition
        sufficient_decrease = f_new <= f_x + c1 * alpha * grad_dot_dir
        done = done | sufficient_decrease

        if done.all():
            break

        # Reduce alpha where condition not met
        alpha = torch.where(done, alpha, alpha * rho)

    return alpha


def _strong_wolfe_line_search(
    f_and_grad: Callable[[Tensor], Tuple[Tensor, Tensor]],
    x: Tensor,
    direction: Tensor,
    f_x: Tensor,
    grad_x: Tensor,
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 25,
    alpha_max: float = 2.5,
) -> Tensor:
    """Line search satisfying the strong Wolfe conditions.

    Finds ``alpha`` such that:

    .. math::

        f(x + \\alpha d) &\\leq f(x) + c_1 \\alpha \\nabla f(x)^T d

        |\\nabla f(x + \\alpha d)^T d| &\\leq c_2 |\\nabla f(x)^T d|

    Implements Algorithm 3.5 (bracketing) and 3.6 (zoom) from
    Nocedal & Wright, "Numerical Optimization", 2nd edition.

    Parameters
    ----------
    f_and_grad : Callable[[Tensor], Tuple[Tensor, Tensor]]
        Function returning ``(f(x), grad_f(x))``.
    x : Tensor
        Current iterate of shape ``(n,)``.
    direction : Tensor
        Search direction of shape ``(n,)``.
    f_x : Tensor
        Function value at ``x``.
    grad_x : Tensor
        Gradient at ``x`` of shape ``(n,)``.
    c1 : float
        Sufficient decrease parameter. Default: 1e-4.
    c2 : float
        Curvature condition parameter. Default: 0.9.
    max_iter : int
        Maximum number of iterations. Default: 25.
    alpha_max : float
        Maximum step length. Default: 2.5.

    Returns
    -------
    Tensor
        Step length ``alpha`` (scalar tensor).
    """
    dg0 = torch.dot(grad_x.flatten(), direction.flatten())
    abs_dg0 = torch.abs(dg0)

    def phi(alpha: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate f and directional derivative at x + alpha * d."""
        x_new = x + alpha * direction
        f_val, g_val = f_and_grad(x_new)
        dg = torch.dot(g_val.flatten(), direction.flatten())
        return f_val, dg

    alpha_prev = torch.zeros((), dtype=x.dtype, device=x.device)
    alpha_curr = torch.ones((), dtype=x.dtype, device=x.device)
    f_prev = f_x.clone()
    dg_prev = dg0.clone()

    # Phase 1: Bracketing (Algorithm 3.5)
    for i in range(max_iter):
        f_curr, dg_curr = phi(alpha_curr)

        # Condition 1: Armijo violated or function increased
        if f_curr > f_x + c1 * alpha_curr * dg0 or (
            f_curr >= f_prev and i > 0
        ):
            return _zoom(
                phi,
                alpha_prev,
                alpha_curr,
                f_prev,
                f_curr,
                dg_prev,
                dg_curr,
                f_x,
                dg0,
                c1,
                c2,
                abs_dg0,
                max_iter,
            )

        # Condition 2: Curvature condition satisfied
        if torch.abs(dg_curr) <= c2 * abs_dg0:
            return alpha_curr

        # Condition 3: Positive slope means bracket found
        if dg_curr >= 0:
            return _zoom(
                phi,
                alpha_curr,
                alpha_prev,
                f_curr,
                f_prev,
                dg_curr,
                dg_prev,
                f_x,
                dg0,
                c1,
                c2,
                abs_dg0,
                max_iter,
            )

        # Expand interval
        alpha_prev = alpha_curr
        f_prev = f_curr
        dg_prev = dg_curr
        alpha_curr = torch.clamp(
            alpha_curr * 2.0,
            max=alpha_max,
        )

    return alpha_curr


def _zoom(
    phi: Callable[[Tensor], Tuple[Tensor, Tensor]],
    alpha_lo: Tensor,
    alpha_hi: Tensor,
    f_lo: Tensor,
    f_hi: Tensor,
    dg_lo: Tensor,
    dg_hi: Tensor,
    f_0: Tensor,
    dg_0: Tensor,
    c1: float,
    c2: float,
    abs_dg0: Tensor,
    max_iter: int,
) -> Tensor:
    """Zoom phase (Algorithm 3.6) for strong Wolfe line search.

    Refines a bracket ``[alpha_lo, alpha_hi]`` to find a step length
    satisfying the strong Wolfe conditions.
    """
    for _ in range(max_iter):
        # Cubic interpolation to find trial step
        alpha_j = _cubic_interpolate(
            alpha_lo,
            f_lo,
            dg_lo,
            alpha_hi,
            f_hi,
            dg_hi,
        )

        # Safeguard: ensure alpha_j is within bracket
        lo = torch.min(alpha_lo, alpha_hi)
        hi = torch.max(alpha_lo, alpha_hi)
        alpha_j = torch.clamp(
            alpha_j, min=lo + 0.1 * (hi - lo), max=hi - 0.1 * (hi - lo)
        )

        f_j, dg_j = phi(alpha_j)

        if f_j > f_0 + c1 * alpha_j * dg_0 or f_j >= f_lo:
            alpha_hi = alpha_j
            f_hi = f_j
            dg_hi = dg_j
        else:
            if torch.abs(dg_j) <= c2 * abs_dg0:
                return alpha_j

            if dg_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
                f_hi = f_lo
                dg_hi = dg_lo

            alpha_lo = alpha_j
            f_lo = f_j
            dg_lo = dg_j

    return alpha_lo

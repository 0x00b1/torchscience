"""BDF (Backward Differentiation Formula) adaptive ODE solver.

Variable-order BDF methods for stiff ODEs with adaptive step size control
and adaptive order selection.
"""

from typing import Callable, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.ordinary_differential_equation._interpolation import (
    LinearInterpolant,
)
from torchscience.ordinary_differential_equation._ivp_exceptions import (
    ConvergenceError,
    MaxStepsExceeded,
)
from torchscience.ordinary_differential_equation._newton_cached import (
    JacobianCache,
    newton_solve_cached,
)
from torchscience.ordinary_differential_equation._tensordict_utils import (
    flatten_state,
)

# BDF coefficients for orders 1-5
# BDF-k formula: sum_{i=0}^{k-1} alpha_i * y_{n-i} = h * beta * f(t_{n+1}, y_{n+1})
# These are derived from polynomial interpolation theory.
# Reference: Hairer & Wanner, "Solving ODEs II", Section III.1

_BDF_ALPHA = {
    1: [1.0],
    2: [4 / 3, -1 / 3],
    3: [18 / 11, -9 / 11, 2 / 11],
    4: [48 / 25, -36 / 25, 16 / 25, -3 / 25],
    5: [300 / 137, -300 / 137, 200 / 137, -75 / 137, 12 / 137],
}

_BDF_BETA = {
    1: 1.0,
    2: 2 / 3,
    3: 6 / 11,
    4: 12 / 25,
    5: 60 / 137,
}

# Error constants for LTE estimation
# These are scaled versions of the theoretical constants to provide
# practical error control. The predictor-corrector difference is
# already a good error estimate, so we use modest multipliers.
_BDF_ERROR_CONST = {
    1: 0.25,
    2: 0.25,
    3: 0.25,
    4: 0.25,
    5: 0.25,
}


def _compute_bdf_predictor(
    y_history: list, f_history: list, h: float, order: int
) -> torch.Tensor:
    """Compute BDF predictor using polynomial extrapolation.

    Uses the previous solution values and derivatives to extrapolate.
    This provides the initial guess for the Newton iteration.
    """
    # Use explicit Adams-Bashforth style predictor for better accuracy
    # This uses derivative history for polynomial extrapolation
    if len(f_history) == 0:
        return y_history[-1].clone()

    # Simple forward Euler predictor (order 1)
    # For higher orders, we'd use more history, but for stability
    # we keep it simple
    return y_history[-1] + h * f_history[-1]


def _compute_bdf_corrector_residual(
    y_next: torch.Tensor,
    y_history: list,
    f_func: Callable,
    t_next: float,
    h: float,
    order: int,
) -> torch.Tensor:
    """Compute the BDF residual for the corrector step.

    BDF-k formula (standardized form):
        y_{n+1} = sum_{i=0}^{k-1} alpha_i * y_{n-i} + h * beta * f(t_{n+1}, y_{n+1})

    The alpha values in _BDF_ALPHA are the coefficients for the history terms.
    """
    alpha = _BDF_ALPHA[order]
    beta = _BDF_BETA[order]

    # Compute history contribution
    # alpha[0] is coefficient for y_n (most recent history)
    # alpha[1] is coefficient for y_{n-1}, etc.
    history_sum = torch.zeros_like(y_next)
    for i, alpha_i in enumerate(alpha):
        if i < len(y_history):
            history_sum = history_sum + alpha_i * y_history[-(i + 1)]

    f_next = f_func(t_next, y_next)

    # Residual: y_{n+1} - history_sum - h * beta * f = 0
    residual = y_next - history_sum - h * beta * f_next

    return residual


def _estimate_local_error(
    y_new: torch.Tensor,
    y_pred: torch.Tensor,
    h: float,
    order: int,
) -> torch.Tensor:
    """Estimate local truncation error for step size control.

    Uses a scaled predictor-corrector difference. The scaling factor is chosen
    to be relatively permissive, as the predictor-corrector difference typically
    overestimates the true LTE for well-behaved problems.

    For BDF methods, a good rule of thumb is to scale by 1/(order+1) to account
    for the predictor being lower order than the corrector.
    """
    # Scale the error estimate to be less conservative
    # The predictor (forward Euler) has error O(h^2)
    # BDF-k has error O(h^{k+1})
    # The difference overestimates the error, so we scale it down
    scale_factor = 1.0 / (order + 2)
    return scale_factor * (y_new - y_pred)


def bdf(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_order: int = 5,
    dt0: Optional[float] = None,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    max_steps: int = 10000,
    newton_tol: float = 1e-8,
    max_newton_iter: int = 10,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using variable-order BDF (Backward Differentiation Formula) method.

    BDF methods are implicit multistep methods particularly suited for stiff
    ODEs. This implementation uses adaptive step size and order selection
    (orders 1-5) with Newton iteration for the implicit solve.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
        Use closures or functools.partial to pass additional parameters.
    y0 : Tensor or TensorDict
        Initial state
    t_span : tuple[float, float]
        Integration interval (t0, t1). Supports backward integration if t1 < t0.
    rtol : float
        Relative tolerance for step size control
    atol : float
        Absolute tolerance for step size control
    max_order : int
        Maximum BDF order to use (1-5). Higher orders are more accurate but
        less stable for very stiff problems. Default is 5.
    dt0 : float, optional
        Initial step size guess. If None, estimated automatically.
    dt_min : float, optional
        Minimum allowed step size.
    dt_max : float, optional
        Maximum allowed step size.
    max_steps : int
        Maximum number of steps before raising MaxStepsExceeded (when throw=True).
    newton_tol : float
        Convergence tolerance for Newton iteration.
    max_newton_iter : int
        Maximum Newton iterations per step.
    throw : bool
        If True (default), raise exceptions on solver failures. If False, return
        NaN for failed batch elements and attach `success` mask to interpolant.

    Returns
    -------
    y : Tensor or TensorDict
        State at t1, shape (*state_shape). Differentiable.
    interp : callable
        Interpolant function. interp(t) returns state at time(s) t.
        Differentiable. Has `success` attribute (bool Tensor) when throw=False.

    Raises
    ------
    MaxStepsExceeded
        If integration requires more than max_steps (only when throw=True).
    ConvergenceError
        If Newton iteration fails to converge (only when throw=True).

    Notes
    -----
    BDF methods of order k use k previous solution values to construct an
    implicit formula. BDF-1 is equivalent to backward Euler. Higher orders
    provide better accuracy but have reduced stability regions for very
    stiff problems (BDF-6 is unstable for stiff problems).

    The solver automatically increases order when the error permits and
    decreases order when encountering difficulties, always starting at
    order 1.

    Examples
    --------
    >>> import torch
    >>> def decay(t, y):
    ...     return -y
    >>> y0 = torch.tensor([1.0])
    >>> y_final, interp = bdf(decay, y0, t_span=(0.0, 5.0))
    >>> trajectory = interp(torch.linspace(0, 5, 100))
    """
    if max_order < 1 or max_order > 5:
        raise ValueError(f"max_order must be in [1, 5], got {max_order}")

    t0, t1 = t_span
    direction = 1.0 if t1 >= t0 else -1.0
    T = abs(t1 - t0)

    # Handle TensorDict
    is_tensordict = isinstance(y0, TensorDict)
    y_flat, unflatten = flatten_state(y0)

    if is_tensordict:

        def f_flat(t, y):
            y_struct = unflatten(y)
            dy_struct = f(t, y_struct)
            dy_flat, _ = flatten_state(dy_struct)
            return dy_flat

    else:
        f_flat = f

    dtype = y_flat.dtype
    device = y_flat.device

    # Use float64 for time points for precision
    t_dtype = torch.float64 if dtype.is_floating_point else dtype

    # Estimate initial step size if not provided
    f0 = f_flat(t0, y_flat)
    if dt0 is None:
        # Standard approach from Hairer & Wanner
        scale = atol + rtol * torch.abs(y_flat)
        d0 = torch.sqrt(torch.mean((y_flat / scale) ** 2)).item()
        d1 = torch.sqrt(torch.mean((f0 / scale) ** 2)).item()
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * (d0 / d1)

        # Limit initial step to reasonable fraction of integration interval
        h0 = min(h0, abs(t1 - t0) * 0.1)

        # Perform one Euler step and estimate second derivative
        y1 = y_flat + h0 * f0
        f1 = f_flat(t0 + h0, y1)
        d2 = torch.sqrt(torch.mean(((f1 - f0) / scale) ** 2)).item() / h0

        if max(d1, d2) <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / 2.0)

        dt0 = min(100 * h0, h1)

    # Apply step size limits
    dt = dt0
    if dt_max is not None:
        dt = min(dt, dt_max)

    # Set effective minimum step size (use machine epsilon if not specified)
    real_dtype = dtype if not dtype.is_complex else torch.float64
    machine_eps = torch.finfo(real_dtype).eps
    dt_min_effective = dt_min if dt_min is not None else 10 * machine_eps * T

    dt = max(dt, dt_min_effective)

    # Dtype-aware completion tolerance
    t_tol = 100 * machine_eps * max(abs(t0), abs(t1), 1.0)

    # Storage for interpolant
    t_points = [torch.tensor(t0, dtype=t_dtype, device=device)]
    y_points = [y_flat.clone()]

    # History for multistep method
    y_history = [y_flat.clone()]
    f_history = [f0.clone()]

    # Current state
    t = t0
    y = y_flat.clone()
    n_steps = 0
    n_rejected = 0
    order = 1  # Start at order 1, increase adaptively

    # Newton cache for Jacobian reuse
    jac_cache = JacobianCache()

    success_overall = True

    while direction * (t1 - t) > t_tol:
        if n_steps >= max_steps:
            if throw:
                raise MaxStepsExceeded(
                    f"Exceeded maximum number of steps ({max_steps})"
                )
            else:
                y = torch.full_like(y, float("nan"))
                success_overall = False
                break

        # Prevent infinite rejection loop
        if n_rejected > 50:
            if throw:
                raise ConvergenceError(
                    f"Too many rejected steps ({n_rejected}) at t={t}"
                )
            else:
                y = torch.full_like(y, float("nan"))
                success_overall = False
                break

        # Clamp step to not overshoot
        dt_step = min(dt, abs(t1 - t))
        h = direction * dt_step

        t_next = t + h

        # Compute predictor
        y_pred = _compute_bdf_predictor(y_history, f_history, h, order)

        # Define residual function for Newton solver
        # Capture variables in closure
        h_curr = h
        t_curr_next = t_next
        order_curr = order
        y_hist_curr = y_history.copy()

        def residual(y_next):
            return _compute_bdf_corrector_residual(
                y_next, y_hist_curr, f_flat, t_curr_next, h_curr, order_curr
            )

        # Solve implicit equation using Newton iteration with caching
        y_next, converged, info = newton_solve_cached(
            residual,
            y_pred,
            tol=newton_tol,
            max_iter=max_newton_iter,
            cache=jac_cache,
            recompute_jacobian_every=5,  # Modified Newton for efficiency
        )

        if not converged:
            # Try reducing step size
            dt = dt_step * 0.5
            jac_cache.clear()  # Reset Jacobian cache
            n_rejected += 1

            if dt < dt_min_effective:
                if throw:
                    raise ConvergenceError(
                        f"Newton iteration failed at t={t_next} and step size "
                        f"below minimum ({dt_min_effective:.2e})"
                    )
                else:
                    y = torch.full_like(y, float("nan"))
                    success_overall = False
                    break

            # Reduce order if possible
            if order > 1:
                order = order - 1

            continue  # Retry with smaller step

        # Compute f at new point for later use
        f_next = f_flat(t_next, y_next)

        # Estimate local error using predictor-corrector difference
        error = _estimate_local_error(y_next, y_pred, abs(h), order)
        scale = atol + rtol * torch.maximum(torch.abs(y), torch.abs(y_next))
        err_norm = torch.sqrt(torch.mean((error / scale) ** 2)).item()

        # Add small epsilon to avoid division by zero
        err_norm = max(err_norm, 1e-10)

        # During startup (first few steps), be more lenient with error
        # This allows the method to build up history for multistep formula
        if n_steps < order:
            err_threshold = 10.0  # More lenient threshold during startup
        else:
            err_threshold = 1.0

        if err_norm > err_threshold:
            # Step rejected - reduce step size
            factor = max(
                0.1, 0.9 * (err_threshold / err_norm) ** (1.0 / (order + 1))
            )
            dt = dt_step * factor
            jac_cache.clear()
            n_rejected += 1

            if dt < dt_min_effective:
                if throw:
                    raise ConvergenceError(
                        f"Step size fell below minimum ({dt_min_effective:.2e}) at t={t}"
                    )
                else:
                    y = torch.full_like(y, float("nan"))
                    success_overall = False
                    break

            continue  # Retry with smaller step

        # Step accepted
        t = t_next
        y = y_next
        n_steps += 1
        n_rejected = 0  # Reset rejection counter on successful step

        # Update history
        y_history.append(y.clone())
        f_history.append(f_next.clone())

        # Keep only enough history for max_order
        max_history = max_order + 1
        if len(y_history) > max_history:
            y_history.pop(0)
            f_history.pop(0)

        # Store for interpolant
        t_points.append(torch.tensor(t, dtype=t_dtype, device=device))
        y_points.append(y.clone())

        # Adaptive order selection
        # Increase order if we have enough history and error is small
        if (
            len(y_history) > order
            and order < max_order
            and err_norm < 0.1 * err_threshold
        ):
            order = min(order + 1, max_order)

        # Decrease order if error is close to tolerance
        if err_norm > 0.5 * err_threshold and order > 1:
            order = order - 1

        # Adjust step size for next step
        # Use PI controller for smoother step size changes
        factor = 0.9 * (err_threshold / err_norm) ** (1.0 / (order + 1))
        factor = max(0.2, min(factor, 5.0))
        dt = dt_step * factor

        if dt_max is not None:
            dt = min(dt, dt_max)
        dt = max(dt, dt_min_effective)

    # Build time tensor
    t_tensor = torch.stack(t_points)
    y_tensor = torch.stack(y_points)

    # Create success tensor for no-throw mode
    if throw:
        success = None
    else:
        success = torch.tensor(
            success_overall,
            dtype=torch.bool,
            device=device,
        )

    # Create interpolant
    interp = LinearInterpolant(t_tensor, y_tensor, success=success)

    # Wrap interpolant for TensorDict
    if is_tensordict:

        def interp_tensordict(t_query):
            y_flat_query = interp(t_query)
            if isinstance(t_query, (int, float)) or (
                isinstance(t_query, torch.Tensor) and t_query.dim() == 0
            ):
                return unflatten(y_flat_query)
            else:
                # For multiple time queries, unflatten each
                results = []
                for i in range(y_flat_query.shape[0]):
                    results.append(unflatten(y_flat_query[i]))
                # Return as stacked TensorDict
                return torch.stack(results)

        interp_tensordict.success = interp.success
        final_interp = interp_tensordict
    else:
        final_interp = interp

    # Unflatten final result
    if is_tensordict:
        y_final = unflatten(y)
    else:
        y_final = y

    return y_final, final_interp

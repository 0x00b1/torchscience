"""Radau IIA (3-stage, order 5) adaptive ODE solver.

Implicit Runge-Kutta method for stiff ODEs with L-stability and
adaptive step size control.
"""

from typing import Callable, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration._interpolation import LinearInterpolant
from torchscience.integration._ivp_exceptions import (
    ConvergenceError,
    MaxStepsExceeded,
)
from torchscience.integration._newton_cached import (
    JacobianCache,
    newton_solve_cached,
)
from torchscience.integration._tensordict_utils import flatten_state

# Radau IIA Butcher tableau (3-stage, order 5)
# Reference: Hairer & Wanner, "Solving ODEs II", Section IV.8
_SQRT6 = 2.449489742783178

# Stage times (c coefficients)
_C = torch.tensor(
    [
        (4 - _SQRT6) / 10,
        (4 + _SQRT6) / 10,
        1.0,
    ],
    dtype=torch.float64,
)

# A matrix (3x3) - implicit RK coefficients
_A = torch.tensor(
    [
        [
            (88 - 7 * _SQRT6) / 360,
            (296 - 169 * _SQRT6) / 1800,
            (-2 + 3 * _SQRT6) / 225,
        ],
        [
            (296 + 169 * _SQRT6) / 1800,
            (88 + 7 * _SQRT6) / 360,
            (-2 - 3 * _SQRT6) / 225,
        ],
        [(16 - _SQRT6) / 36, (16 + _SQRT6) / 36, 1 / 9],
    ],
    dtype=torch.float64,
)

# b weights (same as last row of A for Radau IIA - stiffly accurate)
_B = torch.tensor(
    [(16 - _SQRT6) / 36, (16 + _SQRT6) / 36, 1 / 9], dtype=torch.float64
)

# Embedded error estimator weights (lower order for error estimation)
# These are derived from the embedded method theory for Radau IIA
# We use the difference between y_{n+1} and a lower-order approximation
_B_HAT = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

# Error estimation coefficients: E = h * sum(e_i * k_i)
# where e = b - b_hat gives the local error principal term
_E = _B - _B_HAT


def _compute_radau_residual(
    Z_flat: torch.Tensor,
    y: torch.Tensor,
    f: Callable,
    t: float,
    h: float,
    n: int,
    A: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Compute residual for Radau IIA stage equations.

    The Radau IIA method requires solving for all 3 stages simultaneously:
        Z_i = h * f(t + c_i*h, y + sum_j A_{ij} * Z_j)  for i = 1,2,3

    where Z_i = h * k_i are the scaled stage derivatives.

    Parameters
    ----------
    Z_flat : Tensor
        Flattened stage values, shape (3*n,) where n is state dimension.
    y : Tensor
        Current state, shape (n,).
    f : callable
        Dynamics function f(t, y) -> dy/dt.
    t : float
        Current time.
    h : float
        Step size.
    n : int
        State dimension.
    A : Tensor
        Radau IIA A matrix, shape (3, 3).
    C : Tensor
        Radau IIA c vector, shape (3,).

    Returns
    -------
    residual : Tensor
        Residual vector, shape (3*n,).
    """
    # Reshape Z from (3*n,) to (3, n)
    Z = Z_flat.view(3, n)

    residual_parts = []
    for i in range(3):
        # Compute y + sum_j A[i,j] * Z_j
        y_stage = y.clone()
        for j in range(3):
            y_stage = y_stage + A[i, j] * Z[j]

        # Compute f at stage point
        t_stage = t + C[i].item() * h
        f_stage = f(t_stage, y_stage)

        # Residual: Z_i - h * f(t + c_i*h, y + sum_j A_{ij} * Z_j) = 0
        residual_i = Z[i] - h * f_stage
        residual_parts.append(residual_i)

    return torch.cat(residual_parts)


def _estimate_initial_step(
    f: Callable,
    t0: float,
    y0: torch.Tensor,
    f0: torch.Tensor,
    t1: float,
    rtol: float,
    atol: float,
) -> float:
    """Estimate initial step size using Hairer & Wanner algorithm.

    Reference: Hairer, Norsett, Wanner - "Solving ODEs I", Section II.4
    """
    scale = atol + rtol * torch.abs(y0)

    d0 = torch.sqrt(torch.mean((y0 / scale) ** 2)).item()
    d1 = torch.sqrt(torch.mean((f0 / scale) ** 2)).item()

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * (d0 / d1)

    # Limit initial step to reasonable fraction of integration interval
    h0 = min(h0, abs(t1 - t0) * 0.1)

    # Perform one Euler step and estimate second derivative
    y1 = y0 + h0 * f0
    f1 = f(t0 + h0, y1)
    d2 = torch.sqrt(torch.mean(((f1 - f0) / scale) ** 2)).item() / h0

    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        # Order 5 method
        h1 = (0.01 / max(d1, d2)) ** (1.0 / 5.0)

    return min(100 * h0, h1)


def radau(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    dt0: Optional[float] = None,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    max_steps: int = 10000,
    newton_tol: float = 1e-8,
    max_newton_iter: int = 15,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using Radau IIA method (3-stage, order 5).

    Radau IIA is an implicit Runge-Kutta method with L-stability, making it
    particularly suitable for stiff ODEs. It solves for all 3 stages
    simultaneously using Newton iteration.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
        Use closures or functools.partial to pass additional parameters.
    y0 : Tensor or TensorDict
        Initial state.
    t_span : tuple[float, float]
        Integration interval (t0, t1). Supports backward integration if t1 < t0.
    rtol : float
        Relative tolerance for step size control.
    atol : float
        Absolute tolerance for step size control.
    dt0 : float, optional
        Initial step size guess. If None, estimated automatically.
    dt_min : float, optional
        Minimum allowed step size.
    dt_max : float, optional
        Maximum allowed step size.
    max_steps : int
        Maximum number of steps before raising MaxStepsExceeded.
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
    Radau IIA is a collocation method that uses Radau quadrature points.
    The 3-stage variant achieves order 5 accuracy and is stiffly accurate
    (the numerical solution equals the collocation polynomial at t_{n+1}).

    The method is L-stable, meaning it has optimal damping of stiff components.
    This makes it superior to BDF methods for problems with very large
    negative eigenvalues.

    Examples
    --------
    >>> import torch
    >>> def stiff_decay(t, y):
    ...     return -1000 * y
    >>> y0 = torch.tensor([1.0])
    >>> y_final, interp = radau(stiff_decay, y0, t_span=(0.0, 0.01))
    """
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
    n = y_flat.numel()

    # Use float64 for time points for precision
    t_dtype = torch.float64 if dtype.is_floating_point else dtype

    # Move Butcher tableau to correct device/dtype
    A = _A.to(device=device, dtype=dtype)
    B = _B.to(device=device, dtype=dtype)
    C = _C.to(device=device, dtype=dtype)
    E = _E.to(device=device, dtype=dtype)

    # Estimate initial step size if not provided
    f0 = f_flat(t0, y_flat)
    if dt0 is None:
        dt0 = _estimate_initial_step(f_flat, t0, y_flat, f0, t1, rtol, atol)

    # Apply step size limits
    dt = dt0
    if dt_max is not None:
        dt = min(dt, dt_max)

    # Set effective minimum step size
    real_dtype = dtype if not dtype.is_complex else torch.float64
    machine_eps = torch.finfo(real_dtype).eps
    dt_min_effective = dt_min if dt_min is not None else 10 * machine_eps * T

    dt = max(dt, dt_min_effective)

    # Dtype-aware completion tolerance
    t_tol = 100 * machine_eps * max(abs(t0), abs(t1), 1.0)

    # Storage for interpolant
    t_points = [torch.tensor(t0, dtype=t_dtype, device=device)]
    y_points = [y_flat.clone()]

    # Current state
    t = t0
    y = y_flat.clone()
    n_steps = 0
    n_rejected = 0

    # Newton cache for Jacobian reuse
    jac_cache = JacobianCache()

    success_overall = True

    # Initial guess for Z (stage derivatives scaled by h)
    Z_guess = torch.zeros(3 * n, dtype=dtype, device=device)

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

        # Define residual function for Newton solver
        # Capture current values in closure
        h_curr = h
        t_curr = t
        y_curr = y.clone()

        def residual(Z_flat):
            return _compute_radau_residual(
                Z_flat, y_curr, f_flat, t_curr, h_curr, n, A, C
            )

        # Solve for all stages simultaneously using Newton iteration
        Z_solution, converged, info = newton_solve_cached(
            residual,
            Z_guess,
            tol=newton_tol,
            max_iter=max_newton_iter,
            cache=jac_cache,
            recompute_jacobian_every=5,
        )

        if not converged:
            # Try reducing step size
            dt = dt_step * 0.5
            jac_cache.clear()
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

            # Reset Z_guess on failure
            Z_guess = torch.zeros(3 * n, dtype=dtype, device=device)
            continue

        # Extract stage values
        Z = Z_solution.view(3, n)
        k = Z / h  # Stage derivatives (unscaled)

        # Compute y_{n+1} using Radau IIA (stiffly accurate: y_new = y + Z_3)
        # Since B = last row of A, we have y_new = y + sum(B_i * Z_i)
        y_next = y.clone()
        for i in range(3):
            y_next = y_next + B[i] * Z[i]

        # Error estimation using embedded method
        # Error ~ h * sum(E_i * k_i) where E = B - B_hat
        error = torch.zeros_like(y)
        for i in range(3):
            error = error + E[i] * Z[i]

        # Scale error for comparison with tolerance
        scale = atol + rtol * torch.maximum(torch.abs(y), torch.abs(y_next))
        err_norm = torch.sqrt(torch.mean((error / scale) ** 2)).item()

        # Add small epsilon to avoid division by zero
        err_norm = max(err_norm, 1e-10)

        if err_norm > 1.0:
            # Step rejected - reduce step size
            # Use optimal step size formula for order 5 method
            factor = max(0.1, 0.9 * (1.0 / err_norm) ** (1.0 / 5.0))
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

            # Reset Z_guess on rejection
            Z_guess = torch.zeros(3 * n, dtype=dtype, device=device)
            continue

        # Step accepted
        t = t_next
        y = y_next
        n_steps += 1
        n_rejected = 0

        # Store for interpolant
        t_points.append(torch.tensor(t, dtype=t_dtype, device=device))
        y_points.append(y.clone())

        # Update Z_guess for next step (use scaled values from this step)
        Z_guess = Z_solution.clone()

        # Adjust step size for next step
        # Use PI controller for smoother step size changes
        factor = 0.9 * (1.0 / err_norm) ** (1.0 / 5.0)
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

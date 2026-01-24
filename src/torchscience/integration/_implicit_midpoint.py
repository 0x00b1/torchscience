"""Implicit Midpoint integrator - symplectic AND A-stable."""

from typing import Callable, Tuple, Union

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


def implicit_midpoint(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    dt: float,
    newton_tol: float = 1e-8,
    max_newton_iter: int = 10,
    max_steps: int = 100000,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using implicit midpoint method.

    This is a 2nd-order method that is both symplectic (preserves
    Hamiltonian structure) AND A-stable (handles stiff problems).
    This rare combination makes it ideal for stiff Hamiltonian systems.

    The algorithm:
        y_{n+1} = y_n + h * f(t_n + h/2, (y_n + y_{n+1})/2)

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
    y0 : Tensor or TensorDict
        Initial state.
    t_span : tuple[float, float]
        Integration interval (t0, t1).
    dt : float
        Fixed step size (always positive).
    newton_tol : float
        Convergence tolerance for Newton iteration.
    max_newton_iter : int
        Maximum Newton iterations per step.
    max_steps : int
        Maximum number of steps.
    throw : bool
        If True (default), raise exceptions on failures.

    Returns
    -------
    y : Tensor or TensorDict
        State at t1.
    interp : callable
        Interpolant function.

    Raises
    ------
    ConvergenceError
        If Newton iteration fails to converge (only when throw=True).
    MaxStepsExceeded
        If integration exceeds max_steps (only when throw=True).

    Notes
    -----
    Unlike explicit symplectic integrators (Verlet, Yoshida), implicit
    midpoint can handle non-separable Hamiltonians and stiff systems.
    The trade-off is the cost of Newton iteration per step.

    For separable, non-stiff Hamiltonians, prefer stormer_verlet or yoshida4.
    """
    t0, t1 = t_span
    direction = 1.0 if t1 >= t0 else -1.0
    h = direction * abs(dt)

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

    # Storage for interpolant
    t_points = [torch.tensor(t0, dtype=torch.float64, device=device)]
    y_points = [y_flat.clone()]

    t = t0
    y = y_flat.clone()
    n_steps = 0
    cache = JacobianCache()
    success_overall = True

    while direction * (t1 - t) > 1e-10:
        if n_steps >= max_steps:
            if throw:
                raise MaxStepsExceeded(f"Exceeded maximum steps ({max_steps})")
            else:
                y = torch.full_like(y, float("nan"))
                success_overall = False
                break

        h_step = h
        if direction * (t + h_step - t1) > 0:
            h_step = t1 - t

        t_mid = t + h_step / 2

        # Implicit midpoint: y_{n+1} = y_n + h * f(t_mid, (y_n + y_{n+1})/2)
        # Solve: g(y_{n+1}) = y_{n+1} - y_n - h * f(t_mid, (y_n + y_{n+1})/2) = 0

        # Capture current values in closure
        y_curr = y
        h_curr = h_step
        t_curr_mid = t_mid

        def residual(y_next):
            y_mid = (y_curr + y_next) / 2
            return y_next - y_curr - h_curr * f_flat(t_curr_mid, y_mid)

        # Initial guess: explicit Euler
        y_guess = y + h_step * f_flat(t, y)

        y_next, converged, info = newton_solve_cached(
            residual,
            y_guess,
            tol=newton_tol,
            max_iter=max_newton_iter,
            cache=cache,
            recompute_jacobian_every=1,
        )

        if not converged:
            if throw:
                raise ConvergenceError(
                    f"Newton iteration failed at t={t + h_step} "
                    f"after {info['n_iterations']} iterations"
                )
            else:
                y = torch.full_like(y, float("nan"))
                success_overall = False
                break

        y = y_next
        t = t + h_step

        t_points.append(torch.tensor(t, dtype=torch.float64, device=device))
        y_points.append(y.clone())
        n_steps += 1
        cache.clear()  # Clear cache for next step

    # Build interpolant
    t_tensor = torch.stack(t_points)
    y_tensor = torch.stack(y_points)

    if throw:
        success = None
    else:
        success = torch.tensor(
            success_overall, dtype=torch.bool, device=device
        )

    interp = LinearInterpolant(t_tensor, y_tensor, success=success)

    if is_tensordict:

        def interp_tensordict(t_query):
            y_flat_query = interp(t_query)
            return unflatten(y_flat_query)

        interp_tensordict.success = interp.success
        final_interp = interp_tensordict
        y_final = unflatten(y)
    else:
        final_interp = interp
        y_final = y

    return y_final, final_interp

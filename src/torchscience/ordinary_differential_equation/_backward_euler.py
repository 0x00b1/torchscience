"""Backward Euler (implicit) ODE solver."""

from typing import Callable, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.ordinary_differential_equation._interpolation import (
    LinearInterpolant,
)
from torchscience.ordinary_differential_equation._ivp_exceptions import (
    ConvergenceError,
)
from torchscience.ordinary_differential_equation._newton import newton_solve
from torchscience.ordinary_differential_equation._tensordict_utils import (
    flatten_state,
)


def backward_euler(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    dt: float,
    newton_tol: float = 1e-6,
    max_newton_iter: int = 10,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using backward Euler (implicit) method.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
        Use closures or functools.partial to pass additional parameters.
    y0 : Tensor or TensorDict
        Initial state
    t_span : tuple[float, float]
        Integration interval (t0, t1). Supports backward integration if t1 < t0.
    dt : float
        Fixed step size (always positive; direction inferred from t_span)
    newton_tol : float
        Convergence tolerance for Newton iteration.
    max_newton_iter : int
        Maximum Newton iterations per step. Raises ConvergenceError if exceeded
        (when throw=True).
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
    ConvergenceError
        If Newton iteration fails to converge within max_newton_iter (only when throw=True).

    Notes
    -----
    Jacobian for Newton iteration is computed automatically via torch.func.jacrev.
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

    # Use float64 for time points for precision
    t_dtype = torch.float64 if dtype.is_floating_point else dtype

    t_points = [torch.tensor(t0, dtype=t_dtype, device=device)]
    y_points = [y_flat.clone()]

    t = t0
    y = y_flat.clone()

    success_overall = True

    while direction * (t1 - t) > 1e-10:
        h_step = h
        if direction * (t + h_step - t1) > 0:
            h_step = t1 - t

        t_next = t + h_step

        # Backward Euler: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
        # Solve: g(y_{n+1}) = y_{n+1} - y_n - h * f(t_{n+1}, y_{n+1}) = 0

        # Capture current values in closure
        y_curr = y
        h_curr = h_step
        t_curr_next = t_next

        def residual(y_next):
            return y_next - y_curr - h_curr * f_flat(t_curr_next, y_next)

        # Initial guess: forward Euler prediction
        y_guess = y + h_step * f_flat(t, y)

        # Solve either single system or per-batch element to avoid
        # shape/broadcast issues with captured batch tensors in closures.
        if y.dim() == 1:
            y_next, converged = newton_solve(
                residual, y_guess, tol=newton_tol, max_iter=max_newton_iter
            )
        else:
            B, N = y.shape[0], y.shape[-1]
            y_next_rows = []
            converged_all = True

            for i in range(B):
                y_i = y[i]

                def residual_i(y_next_i):
                    return (
                        y_next_i - y_i - h_curr * f_flat(t_curr_next, y_next_i)
                    )

                y_guess_i = y_guess[i]
                y_next_i, ok = newton_solve(
                    residual_i,
                    y_guess_i,
                    tol=newton_tol,
                    max_iter=max_newton_iter,
                )
                if not ok:
                    converged_all = False
                y_next_rows.append(y_next_i)

            y_next = torch.stack(y_next_rows, dim=0).reshape(B, N)
            converged = converged_all

        if not converged:
            if throw:
                raise ConvergenceError(
                    f"Newton iteration failed to converge at t={t_next} "
                    f"after {max_newton_iter} iterations"
                )
            else:
                y = torch.full_like(y, float("nan"))
                success_overall = False
                break

        y = y_next
        t = t_next

        t_points.append(torch.tensor(t, dtype=t_dtype, device=device))
        y_points.append(y.clone())

    t_tensor = torch.stack(t_points)
    y_tensor = torch.stack(y_points)

    if throw:
        success = None
    else:
        success = torch.tensor(
            success_overall,
            dtype=torch.bool,
            device=device,
        )
        # Expand to batch shape if needed
        if y_flat.dim() > 1:
            success = success.expand(y_flat.shape[:-1])

    interp = LinearInterpolant(t_tensor, y_tensor, success=success)

    if is_tensordict:

        def interp_tensordict(t_query):
            y_flat_query = interp(t_query)
            if isinstance(t_query, (int, float)) or (
                isinstance(t_query, torch.Tensor) and t_query.dim() == 0
            ):
                return unflatten(y_flat_query)
            else:
                return torch.stack(
                    [
                        unflatten(y_flat_query[i])
                        for i in range(y_flat_query.shape[0])
                    ]
                )

        interp_tensordict.success = interp.success
        final_interp = interp_tensordict
    else:
        final_interp = interp

    if is_tensordict:
        y_final = unflatten(y)
    else:
        y_final = y

    return y_final, final_interp

"""Midpoint (explicit) ODE solver."""

from typing import Callable, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.ordinary_differential_equation._interpolation import (
    LinearInterpolant,
)
from torchscience.ordinary_differential_equation._tensordict_utils import (
    flatten_state,
)


def midpoint(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    dt: float,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using explicit midpoint method (2nd order Runge-Kutta).

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

    t_points = [torch.tensor(t0, dtype=dtype, device=device)]
    y_points = [y_flat.clone()]

    t = t0
    y = y_flat.clone()

    while direction * (t1 - t) > 1e-10:
        h_step = h
        if direction * (t + h_step - t1) > 0:
            h_step = t1 - t

        # Midpoint method:
        # k1 = f(t_n, y_n)
        # k2 = f(t_n + h/2, y_n + h/2 * k1)
        # y_{n+1} = y_n + h * k2
        k1 = f_flat(t, y)
        k2 = f_flat(t + h_step / 2, y + h_step / 2 * k1)
        y = y + h_step * k2
        t = t + h_step

        t_points.append(torch.tensor(t, dtype=dtype, device=device))
        y_points.append(y.clone())

    t_tensor = torch.stack(t_points)
    y_tensor = torch.stack(y_points)

    success = (
        None
        if throw
        else torch.ones(
            y_flat.shape[:-1] if y_flat.dim() > 1 else (),
            dtype=torch.bool,
            device=device,
        )
    )

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

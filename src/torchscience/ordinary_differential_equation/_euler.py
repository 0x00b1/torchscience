"""Forward Euler ODE solver."""

from typing import Callable, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.ordinary_differential_equation._interpolation import (
    LinearInterpolant,
)
from torchscience.ordinary_differential_equation._tensordict_utils import (
    flatten_state,
)


def euler(
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
    Solve ODE using forward Euler method.

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

    # Wrap dynamics for flattened state
    if is_tensordict:

        def f_flat(t, y):
            y_struct = unflatten(y)
            dy_struct = f(t, y_struct)
            dy_flat, _ = flatten_state(dy_struct)
            return dy_flat

    else:
        f_flat = f

    # Get dtype and device
    dtype = y_flat.dtype
    device = y_flat.device

    # Storage for interpolant
    t_points = [torch.tensor(t0, dtype=dtype, device=device)]
    y_points = [y_flat.clone()]

    t = t0
    y = y_flat.clone()

    while direction * (t1 - t) > 1e-10:
        # Clamp step to not overshoot
        h_step = h
        if direction * (t + h_step - t1) > 0:
            h_step = t1 - t

        # Euler step: y_{n+1} = y_n + h * f(t_n, y_n)
        dy = f_flat(t, y)
        y = y + h_step * dy
        t = t + h_step

        # Store for interpolant
        t_points.append(torch.tensor(t, dtype=dtype, device=device))
        y_points.append(y.clone())

    # Build interpolant
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

    # Wrap interpolant for TensorDict using vectorized unflatten
    if is_tensordict:

        class TensorDictInterpolant:
            def __init__(self, base_interp, unflatten_fn):
                self._base = base_interp
                self._unflatten = unflatten_fn
                self.success = base_interp.success

            def __call__(self, t_query):
                y_flat_query = self._base(t_query)
                return self._unflatten(y_flat_query)

        final_interp = TensorDictInterpolant(interp, unflatten)
    else:
        final_interp = interp

    # Unflatten final result
    if is_tensordict:
        y_final = unflatten(y)
    else:
        y_final = y

    return y_final, final_interp

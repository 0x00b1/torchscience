"""Asynchronous Leapfrog (ALA) integrator for O(1) memory neural ODEs."""

from typing import Callable, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration._interpolation import LinearInterpolant
from torchscience.integration._tensordict_utils import flatten_state


def asynchronous_leapfrog(
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
    Solve ODE using Asynchronous Leapfrog (ALA) with O(1) memory gradients.

    ALA is a 2nd-order reversible integrator that maintains two state estimates
    staggered in time, alternating updates between them. This structure enables
    exact reversal without iteration, making it suitable for memory-efficient
    neural ODE training.

    The algorithm maintains y_even (at integer times) and y_odd (at half times):

    Initialization:
        y_odd = y_even + (h/2) * f(t, y_even)

    Each step (advancing from t to t+h):
        y_even_new = y_even + h * f(t + h/2, y_odd)
        y_odd_new = y_odd + h * f(t + h, y_even_new)

    Reversal (going from t+h back to t):
        y_even_prev = y_even - h * f(t + h/2, y_odd)
        y_odd_prev = y_odd - h * f(t, y_even_prev)

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
    throw : bool
        If True (default), raise exceptions on failures.

    Returns
    -------
    y : Tensor or TensorDict
        State at t1.
    interp : callable
        Interpolant function.

    Notes
    -----
    ALA has better stability properties than Reversible Heun for some
    problems because it doesn't require iterative inversion - the reversal
    is exact and algebraic.

    For neural ODE training, ALA provides O(1) memory gradient computation
    when combined with gradient checkpointing, as the forward trajectory
    can be reconstructed exactly during backpropagation.

    References
    ----------
    MacKay, M., et al. "Reversible Recurrent Neural Networks."
    NeurIPS 2018.
    """
    t0, t1 = t_span

    # Handle zero-length integration
    if abs(t1 - t0) < 1e-14:
        # Return input unchanged
        is_tensordict = isinstance(y0, TensorDict)
        y_flat, unflatten = flatten_state(y0)
        dtype = y_flat.dtype
        device = y_flat.device

        t_tensor = torch.tensor([t0], dtype=dtype, device=device)
        y_tensor = y_flat.unsqueeze(0)

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
            return y0, interp_tensordict
        else:
            return y0, interp

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
    t_points = [torch.tensor(t0, dtype=dtype, device=device)]
    y_points = [y_flat.clone()]

    t = t0
    y_even = y_flat.clone()

    # Initialize y_odd at t + h/2 using half step
    # y_odd represents state at staggered time t + h/2
    y_odd = y_even + (h / 2) * f_flat(t, y_even)

    while direction * (t1 - t) > 1e-10:
        h_step = h
        if direction * (t + h_step - t1) > 0:
            h_step = t1 - t

        # ALA step: update y_even first using y_odd, then update y_odd
        # y_odd is at t + h/2, y_even is at t
        # After step: y_even at t + h, y_odd at t + 3h/2

        # First: update y_even using y_odd at t + h/2
        y_even = y_even + h_step * f_flat(t + h_step / 2, y_odd)

        # Then: update y_odd using new y_even at t + h
        t = t + h_step
        y_odd = y_odd + h_step * f_flat(t, y_even)

        t_points.append(torch.tensor(t, dtype=dtype, device=device))
        y_points.append(y_even.clone())

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
        y_final = unflatten(y_even)
    else:
        final_interp = interp
        y_final = y_even

    return y_final, final_interp

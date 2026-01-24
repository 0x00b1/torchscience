"""Störmer-Verlet (velocity Verlet) symplectic integrator for Hamiltonian systems."""

from typing import Callable, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.ordinary_differential_equation._interpolation import (
    LinearInterpolant,
)
from torchscience.ordinary_differential_equation._tensordict_utils import (
    flatten_state,
)


def stormer_verlet(
    grad_potential: Callable[
        [Union[float, torch.Tensor], torch.Tensor], torch.Tensor
    ],
    grad_kinetic: Callable[
        [Union[float, torch.Tensor], torch.Tensor], torch.Tensor
    ],
    q0: Union[torch.Tensor, TensorDict],
    p0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    dt: float,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Union[torch.Tensor, TensorDict],
    Callable,
]:
    """
    Solve Hamiltonian system using Störmer-Verlet (velocity Verlet) method.

    This is a 2nd-order symplectic integrator for separable Hamiltonians
    of the form H(q, p) = T(p) + V(q). It preserves the symplectic structure
    exactly, ensuring energy conservation over long integrations.

    The algorithm (velocity Verlet form):
        p_{1/2} = p_n - (h/2) * grad_V(q_n)
        q_{n+1} = q_n + h * grad_T(p_{1/2})
        p_{n+1} = p_{1/2} - (h/2) * grad_V(q_{n+1})

    Parameters
    ----------
    grad_potential : callable
        Gradient of potential energy: grad_V(t, q) -> force.
        For V(q) = q^2/2, this returns q.
    grad_kinetic : callable
        Gradient of kinetic energy: grad_T(t, p) -> velocity.
        For T(p) = p^2/(2m), this returns p/m.
    q0 : Tensor or TensorDict
        Initial position.
    p0 : Tensor or TensorDict
        Initial momentum.
    t_span : tuple[float, float]
        Integration interval (t0, t1). Supports backward integration if t1 < t0.
    dt : float
        Fixed step size (always positive; direction inferred from t_span).
    throw : bool
        If True (default), raise exceptions on failures. If False, attach
        `success` mask to interpolant.

    Returns
    -------
    q : Tensor or TensorDict
        Final position.
    p : Tensor or TensorDict
        Final momentum.
    interp : callable
        Interpolant: interp(t) -> (q, p) at time t.
        Has `success` attribute when throw=False.

    Notes
    -----
    For non-separable Hamiltonians (where T depends on q or V depends on p),
    use implicit_midpoint instead.

    The Störmer-Verlet method is:
    - Symplectic: preserves phase space volume exactly
    - Time-reversible: integrating backward returns to initial state
    - 2nd-order accurate: local error is O(h^3), global error is O(h^2)
    - Energy-conserving: energy oscillates around true value without drift

    Examples
    --------
    Simple harmonic oscillator:

    >>> def grad_V(t, q): return q  # V = q^2/2
    >>> def grad_T(t, p): return p  # T = p^2/2
    >>> q0 = torch.tensor([1.0])
    >>> p0 = torch.tensor([0.0])
    >>> q, p, interp = stormer_verlet(grad_V, grad_T, q0, p0, (0, 2*pi), dt=0.01)
    """
    t0, t1 = t_span
    direction = 1.0 if t1 >= t0 else -1.0
    h = direction * abs(dt)

    # Handle TensorDict for q
    is_tensordict_q = isinstance(q0, TensorDict)
    q_flat, unflatten_q = flatten_state(q0)

    is_tensordict_p = isinstance(p0, TensorDict)
    p_flat, unflatten_p = flatten_state(p0)

    if is_tensordict_q:

        def grad_V_flat(t, q):
            q_struct = unflatten_q(q)
            dV_struct = grad_potential(t, q_struct)
            dV_flat, _ = flatten_state(dV_struct)
            return dV_flat

    else:
        grad_V_flat = grad_potential

    if is_tensordict_p:

        def grad_T_flat(t, p):
            p_struct = unflatten_p(p)
            dT_struct = grad_kinetic(t, p_struct)
            dT_flat, _ = flatten_state(dT_struct)
            return dT_flat

    else:
        grad_T_flat = grad_kinetic

    dtype = q_flat.dtype
    device = q_flat.device

    # Storage for interpolant
    t_points = [torch.tensor(t0, dtype=dtype, device=device)]
    q_points = [q_flat.clone()]
    p_points = [p_flat.clone()]

    t = t0
    q = q_flat.clone()
    p = p_flat.clone()

    while direction * (t1 - t) > 1e-10:
        h_step = h
        if direction * (t + h_step - t1) > 0:
            h_step = t1 - t

        # Velocity Verlet algorithm:
        # 1. Half-step momentum update
        p_half = p - (h_step / 2) * grad_V_flat(t, q)

        # 2. Full-step position update
        q = q + h_step * grad_T_flat(t + h_step / 2, p_half)

        # 3. Half-step momentum update
        t = t + h_step
        p = p_half - (h_step / 2) * grad_V_flat(t, q)

        t_points.append(torch.tensor(t, dtype=dtype, device=device))
        q_points.append(q.clone())
        p_points.append(p.clone())

    # Build interpolants
    t_tensor = torch.stack(t_points)
    q_tensor = torch.stack(q_points)
    p_tensor = torch.stack(p_points)

    success = (
        None if throw else torch.tensor(True, dtype=torch.bool, device=device)
    )

    q_interp = LinearInterpolant(t_tensor, q_tensor, success=success)
    p_interp = LinearInterpolant(t_tensor, p_tensor, success=success)

    def interp(t_query):
        """Interpolate (q, p) at time t_query."""
        q_val = q_interp(t_query)
        p_val = p_interp(t_query)

        # Unflatten if needed
        if is_tensordict_q:
            q_result = unflatten_q(q_val)
        else:
            q_result = q_val

        if is_tensordict_p:
            p_result = unflatten_p(p_val)
        else:
            p_result = p_val

        return q_result, p_result

    interp.success = success

    # Unflatten final results
    if is_tensordict_q:
        q_final = unflatten_q(q)
    else:
        q_final = q

    if is_tensordict_p:
        p_final = unflatten_p(p)
    else:
        p_final = p

    return q_final, p_final, interp

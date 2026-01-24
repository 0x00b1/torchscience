"""Adams-Bashforth-Moulton predictor-corrector multistep method.

This module provides a variable-order (1-5) Adams method that is efficient
when the dynamics function f is expensive to evaluate.
"""

from typing import Callable, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.ordinary_differential_equation._interpolation import (
    LinearInterpolant,
)
from torchscience.ordinary_differential_equation._ivp_exceptions import (
    MaxStepsExceeded,
    StepSizeError,
)
from torchscience.ordinary_differential_equation._tensordict_utils import (
    flatten_state,
)

# Adams-Bashforth coefficients for orders 1-5
# AB_k: y_{n+1} = y_n + h * sum(b_i * f_{n+1-i}) for i=1..k
_AB_COEFFS = {
    1: [1.0],
    2: [3 / 2, -1 / 2],
    3: [23 / 12, -16 / 12, 5 / 12],
    4: [55 / 24, -59 / 24, 37 / 24, -9 / 24],
    5: [1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720],
}

# Adams-Moulton coefficients for orders 1-5
# AM_k: y_{n+1} = y_n + h * sum(b_i * f_{n+2-i}) for i=1..k+1
_AM_COEFFS = {
    1: [1 / 2, 1 / 2],
    2: [5 / 12, 8 / 12, -1 / 12],
    3: [9 / 24, 19 / 24, -5 / 24, 1 / 24],
    4: [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720],
    5: [
        475 / 1440,
        1427 / 1440,
        -798 / 1440,
        482 / 1440,
        -173 / 1440,
        27 / 1440,
    ],
}


def adams(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_order: int = 5,
    max_steps: int = 10000,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using Adams-Bashforth-Moulton predictor-corrector method.

    This is a variable-order multistep method that is efficient when the
    dynamics function f is expensive to evaluate, as it reuses previous
    function evaluations.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
    y0 : Tensor or TensorDict
        Initial state.
    t_span : tuple[float, float]
        Integration interval (t0, t1).
    rtol : float
        Relative tolerance for adaptive stepping. Default: 1e-6.
    atol : float
        Absolute tolerance for adaptive stepping. Default: 1e-8.
    max_order : int
        Maximum order to use (1-5). Default: 5.
    max_steps : int
        Maximum number of steps before raising error. Default: 10000.
    dt_min : float, optional
        Minimum allowed step size. Default: |t1-t0| * 1e-12.
    dt_max : float, optional
        Maximum allowed step size. Default: |t1-t0|.
    throw : bool
        If True (default), raise exceptions on failures.

    Returns
    -------
    y : Tensor or TensorDict
        State at t1.
    interp : callable
        Interpolant function for dense output.

    Notes
    -----
    Adams methods are multistep methods that use history from previous steps.
    They require a startup phase using a single-step method (RK4) to build
    up the required history.

    The predictor (Adams-Bashforth) is explicit, and the corrector
    (Adams-Moulton) is implicit but solved with a single iteration
    (PECE mode: Predict-Evaluate-Correct-Evaluate).

    Adams methods are preferred when:

    - f is expensive to evaluate
    - The problem is smooth (not stiff)
    - High accuracy is needed

    For stiff problems, use bdf() or radau() instead.

    References
    ----------
    Hairer, E., Norsett, S. P., & Wanner, G. (1993).
    Solving Ordinary Differential Equations I: Nonstiff Problems.
    Springer Series in Computational Mathematics.
    """
    t0, t1 = t_span
    max_order = min(max_order, 5)  # Limit to implemented orders

    # Handle zero-length integration
    if abs(t1 - t0) < 1e-14:
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
    t_range = abs(t1 - t0)

    if dt_min is None:
        dt_min = t_range * 1e-12
    if dt_max is None:
        dt_max = t_range

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

    # History storage for multistep method
    f_history = []  # Store f evaluations

    # Initial step size estimate
    f0 = f_flat(t0, y_flat)
    f_history.append(f0)
    scale = atol + rtol * torch.abs(y_flat)

    # Handle complex numbers for norm computation
    y_norm = y_flat / scale
    f_norm = f0 / scale
    if y_norm.is_complex():
        d0 = torch.sqrt(torch.mean(torch.abs(y_norm) ** 2)).real.item()
        d1 = torch.sqrt(torch.mean(torch.abs(f_norm) ** 2)).real.item()
    else:
        d0 = torch.sqrt(torch.mean(y_norm**2)).item()
        d1 = torch.sqrt(torch.mean(f_norm**2)).item()

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    h = min(h0, dt_max) * direction

    t = t0
    y = y_flat.clone()
    n_steps = 0
    order = 1  # Start with order 1

    while direction * (t1 - t) > 1e-10:
        if n_steps >= max_steps:
            if throw:
                raise MaxStepsExceeded(
                    f"Maximum steps ({max_steps}) exceeded at t={t}. "
                    f"Consider increasing max_steps or loosening tolerances."
                )
            break

        # Ensure we don't overshoot
        if direction * (t + h - t1) > 0:
            h = t1 - t

        # Need at least 'order' history points for Adams-Bashforth
        if len(f_history) < order:
            # Startup: use RK4 to build history
            k1 = f_flat(t, y)
            k2 = f_flat(t + h / 2, y + h / 2 * k1)
            k3 = f_flat(t + h / 2, y + h / 2 * k2)
            k4 = f_flat(t + h, y + h * k3)
            y_new = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            t = t + h
            y = y_new
            f_new = f_flat(t, y)
            f_history.append(f_new)

            n_steps += 1
            t_points.append(torch.tensor(t, dtype=dtype, device=device))
            y_points.append(y.clone())

            # Increase order gradually
            if len(f_history) >= order + 1 and order < max_order:
                order += 1

            continue

        # Adams-Bashforth predictor
        ab_coeffs = _AB_COEFFS[order]
        y_pred = y.clone()
        for i, coeff in enumerate(ab_coeffs):
            y_pred = y_pred + h * coeff * f_history[-(i + 1)]

        # Evaluate f at predicted point
        f_pred = f_flat(t + h, y_pred)

        # Adams-Moulton corrector (one iteration - PECE mode)
        am_coeffs = _AM_COEFFS[order]
        y_corr = y.clone()
        y_corr = y_corr + h * am_coeffs[0] * f_pred
        for i, coeff in enumerate(am_coeffs[1:]):
            if i < len(f_history):
                y_corr = y_corr + h * coeff * f_history[-(i + 1)]

        # Error estimate: difference between predictor and corrector
        err = y_corr - y_pred
        scale = atol + rtol * torch.maximum(torch.abs(y), torch.abs(y_corr))

        if err.is_complex():
            err_norm = torch.sqrt(
                torch.mean(torch.abs(err / scale) ** 2)
            ).real.item()
        else:
            err_norm = torch.sqrt(torch.mean((err / scale) ** 2)).item()

        # Step acceptance
        if err_norm <= 1.0:
            # Accept step
            t = t + h
            y = y_corr
            f_new = f_flat(t, y)

            # Update history (keep only what's needed)
            f_history.append(f_new)
            if len(f_history) > max_order + 2:
                f_history.pop(0)

            n_steps += 1
            t_points.append(torch.tensor(t, dtype=dtype, device=device))
            y_points.append(y.clone())

            # Increase order if error is small and we have enough history
            if (
                err_norm < 0.1
                and order < max_order
                and len(f_history) >= order + 2
            ):
                order += 1

            # Increase step size
            if err_norm < 1e-10:
                factor = 2.0
            else:
                factor = min(
                    2.0, 0.9 * (1.0 / err_norm) ** (1.0 / (order + 1))
                )
            h = h * factor
            h = direction * min(abs(h), dt_max)
        else:
            # Reject step, reduce step size
            factor = max(0.1, 0.9 * (1.0 / err_norm) ** (1.0 / (order + 1)))
            h = h * factor

            # Reduce order if struggling
            if order > 1:
                order -= 1

            if abs(h) < dt_min:
                if throw:
                    raise StepSizeError(
                        f"Step size ({abs(h):.2e}) fell below minimum "
                        f"({dt_min:.2e}) at t={t}. "
                        f"Problem may be stiff. Try bdf() or radau() instead."
                    )
                break

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
        return unflatten(y), interp_tensordict
    else:
        return y, interp

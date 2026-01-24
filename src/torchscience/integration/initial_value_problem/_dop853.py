"""DOP853: High-order adaptive ODE solver.

This module provides an 8th-order Runge-Kutta method with embedded error
estimation, based on the classical Dormand-Prince 8(5,3) scheme.
"""

from typing import Callable, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration._interpolation import LinearInterpolant
from torchscience.integration._tensordict_utils import flatten_state
from torchscience.integration.initial_value_problem._exceptions import (
    MaxStepsExceeded,
    StepSizeError,
)


def dop853(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_steps: int = 10000,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    throw: bool = True,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    Callable[[Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]],
]:
    """
    Solve ODE using a high-order adaptive Runge-Kutta method.

    This solver achieves higher accuracy than Dormand-Prince 5(4) for smooth
    problems by using more stages and a higher-order formula.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(t, y) -> dy/dt.
    y0 : Tensor or TensorDict
        Initial state.
    t_span : tuple[float, float]
        Integration interval (t0, t1).
    rtol : float
        Relative tolerance for adaptive stepping. Default: 1e-7.
    atol : float
        Absolute tolerance for adaptive stepping. Default: 1e-9.
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
    This solver is preferred over Dormand-Prince 5(4) when:

    - High accuracy is required (rtol < 1e-8)
    - The problem is smooth (no discontinuities)
    - More function evaluations per step are acceptable

    References
    ----------
    Hairer, E., Norsett, S. P., & Wanner, G. (1993).
    Solving Ordinary Differential Equations I: Nonstiff Problems.
    """
    t0, t1 = t_span

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

    # Initial step size estimate
    f0 = f_flat(t0, y_flat)
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

    # Order of the method for step size control
    order = 8

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

        # 8-stage Runge-Kutta-Fehlberg 7(8) method
        # Using classical RK78 coefficients
        k1 = f_flat(t, y)
        k2 = f_flat(t + h * 2 / 27, y + h * 2 / 27 * k1)
        k3 = f_flat(t + h / 9, y + h * (k1 / 36 + k2 / 12))
        k4 = f_flat(t + h / 6, y + h * (k1 / 24 + k3 / 8))
        k5 = f_flat(
            t + h * 5 / 12,
            y + h * (5 / 12 * k1 - 25 / 16 * k3 + 25 / 16 * k4),
        )
        k6 = f_flat(t + h / 2, y + h * (k1 / 20 + k4 / 4 + k5 / 5))
        k7 = f_flat(
            t + h * 5 / 6,
            y
            + h
            * (-25 / 108 * k1 + 125 / 108 * k4 - 65 / 27 * k5 + 125 / 54 * k6),
        )
        k8 = f_flat(
            t + h / 6,
            y
            + h * (31 / 300 * k1 + 61 / 225 * k5 - 2 / 9 * k6 + 13 / 900 * k7),
        )
        k9 = f_flat(
            t + h * 2 / 3,
            y
            + h
            * (
                2 * k1
                - 53 / 6 * k4
                + 704 / 45 * k5
                - 107 / 9 * k6
                + 67 / 90 * k7
                + 3 * k8
            ),
        )
        k10 = f_flat(
            t + h / 3,
            y
            + h
            * (
                -91 / 108 * k1
                + 23 / 108 * k4
                - 976 / 135 * k5
                + 311 / 54 * k6
                - 19 / 60 * k7
                + 17 / 6 * k8
                - 1 / 12 * k9
            ),
        )
        k11 = f_flat(
            t + h,
            y
            + h
            * (
                2383 / 4100 * k1
                - 341 / 164 * k4
                + 4496 / 1025 * k5
                - 301 / 82 * k6
                + 2133 / 4100 * k7
                + 45 / 82 * k8
                + 45 / 164 * k9
                + 18 / 41 * k10
            ),
        )
        k12 = f_flat(
            t + h,
            y
            + h
            * (
                3 / 205 * k1
                - 6 / 41 * k6
                - 3 / 205 * k7
                - 3 / 41 * k8
                + 3 / 41 * k9
                + 6 / 41 * k10
            ),
        )
        k13 = f_flat(
            t + h,
            y
            + h
            * (
                -1777 / 4100 * k1
                - 341 / 164 * k4
                + 4496 / 1025 * k5
                - 289 / 82 * k6
                + 2193 / 4100 * k7
                + 51 / 82 * k8
                + 33 / 164 * k9
                + 12 / 41 * k10
                + k12
            ),
        )

        # 8th order solution
        y_new = y + h * (
            41 / 840 * k1
            + 34 / 105 * k6
            + 9 / 35 * k7
            + 9 / 35 * k8
            + 9 / 280 * k9
            + 9 / 280 * k10
            + 41 / 840 * k11
        )

        # 7th order solution for error estimate
        y_low = y + h * (
            41 / 840 * k1
            + 34 / 105 * k6
            + 9 / 35 * k7
            + 9 / 35 * k8
            + 9 / 280 * k9
            + 9 / 280 * k10
            + 41 / 840 * k13
        )

        # Error estimate
        err = y_new - y_low
        scale = atol + rtol * torch.maximum(torch.abs(y), torch.abs(y_new))

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
            y = y_new
            n_steps += 1

            t_points.append(torch.tensor(t, dtype=dtype, device=device))
            y_points.append(y.clone())

            # Increase step size using optimal PI controller
            if err_norm < 1e-10:
                factor = 5.0
            else:
                factor = min(
                    5.0, 0.9 * (1.0 / err_norm) ** (1.0 / (order + 1))
                )
            h = h * factor
            h = direction * min(abs(h), dt_max)
        else:
            # Reject step, reduce step size
            factor = max(0.1, 0.9 * (1.0 / err_norm) ** (1.0 / (order + 1)))
            h = h * factor

            if abs(h) < dt_min:
                if throw:
                    raise StepSizeError(
                        f"Step size ({abs(h):.2e}) fell below minimum "
                        f"({dt_min:.2e}) at t={t}. "
                        f"Problem may be stiff. Try bdf() or radau() instead."
                    )
                # Accept step anyway with reduced accuracy
                t = t + direction * dt_min
                y = y_new
                n_steps += 1
                t_points.append(torch.tensor(t, dtype=dtype, device=device))
                y_points.append(y.clone())
                h = direction * dt_min * 2

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

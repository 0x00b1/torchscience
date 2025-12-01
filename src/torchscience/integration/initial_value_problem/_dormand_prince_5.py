"""Dormand-Prince 5(4) adaptive ODE solver."""

import warnings
from functools import lru_cache
from typing import Callable, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration._interpolation import DP5Interpolant
from torchscience.integration._tensordict_utils import flatten_state
from torchscience.integration.initial_value_problem._exceptions import (
    MaxStepsExceeded,
    StepSizeError,
)

# Dormand-Prince 5(4) Butcher tableau coefficients (raw values)
# fmt: off
_C_RAW = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
_A_RAW = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
]
_B5_RAW = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
_B4_RAW = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
# fmt: on


@lru_cache(maxsize=8)
def _get_tableau(dtype_str: str, device_str: str):
    """Get Butcher tableau tensors for the given dtype and device.

    Uses string keys for proper LRU cache hashing.
    """
    # Convert strings back to torch types
    dtype = getattr(torch, dtype_str)
    device = torch.device(device_str)

    C = torch.tensor(_C_RAW, dtype=dtype, device=device)
    A = [
        [torch.tensor(a, dtype=dtype, device=device) for a in row]
        for row in _A_RAW
    ]
    B5 = torch.tensor(_B5_RAW, dtype=dtype, device=device)
    B4 = torch.tensor(_B4_RAW, dtype=dtype, device=device)
    return C, A, B5, B4


def dormand_prince_5(
    f: Callable[[Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
    y0: Union[torch.Tensor, TensorDict],
    t_span: Tuple[float, float],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    dt0: Optional[float] = None,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    max_steps: int = 10000,
    max_segments: Optional[int] = None,
    throw: bool = True,
    error_weights: Optional[torch.Tensor] = None,
) -> Tuple[
    Union[torch.Tensor, TensorDict],
    DP5Interpolant,
]:
    """
    Solve ODE using Dormand-Prince 5(4) adaptive method.

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
    dt0 : float, optional
        Initial step size guess. If None, estimated automatically.
    dt_min : float, optional
        Minimum allowed step size. Raises StepSizeError if step would go below
        (when throw=True).
    dt_max : float, optional
        Maximum allowed step size. Step size is clamped to this value.
    max_steps : int
        Maximum number of steps before raising MaxStepsExceeded (when throw=True).
    max_segments : int, optional
        Maximum number of segments to store for dense output. If exceeded,
        oldest segments are dropped and a warning is issued. Default is None
        (unlimited). Useful for long integrations where memory is a concern.
        Note: For long integrations (10k+ steps), consider using this parameter
        or periodic checkpointing to manage memory.
    throw : bool
        If True (default), raise exceptions on solver failures. If False, return
        NaN for failed batch elements and attach `success` mask to interpolant.
    error_weights : Tensor, optional
        Per-component error weights for seminorm error control. Shape must be
        broadcastable to the state shape. Values > 1 increase sensitivity to
        that component's error (tighter control), while values < 1 decrease
        sensitivity (looser control). Default is None (uniform weighting).

    Returns
    -------
    y : Tensor or TensorDict
        State at t1, shape (*state_shape). Differentiable.
    interp : DP5Interpolant
        Interpolant function. interp(t) returns state at time(s) t.
        Differentiable. Has `success` attribute (bool Tensor) when throw=False.

    Raises
    ------
    MaxStepsExceeded
        If integration requires more than max_steps (only when throw=True).
    StepSizeError
        If adaptive step size falls below dt_min (only when throw=True).

    Notes
    -----
    **Batched Integration**: All batch elements share the same time grid.
    Steps are accepted if ANY element passes the error test. For problems
    with heterogeneous stiffness across batch elements, consider solving
    separately for better per-element accuracy.
    """
    t0, t1 = t_span
    direction = 1.0 if t1 >= t0 else -1.0
    t_end = t1

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

    # Get Butcher tableau (use string keys for proper caching)
    dtype_str = str(dtype).replace("torch.", "")
    device_str = str(device)
    C, A, B5, B4 = _get_tableau(dtype_str, device_str)

    # Validate and convert error_weights if provided
    if error_weights is not None:
        if (error_weights <= 0).any():
            raise ValueError("error_weights must be strictly positive")
        error_weights = error_weights.to(dtype=dtype, device=device)

    # Determine batch shape explicitly
    # For TensorDict: batch_size is explicit
    # For Tensor: assume last dim is state, rest is batch
    if is_tensordict:
        batch_shape = tuple(y0.batch_size)
    else:
        batch_shape = tuple(y_flat.shape[:-1]) if y_flat.dim() > 1 else ()

    # Compute initial k1 (used for both dt0 estimation and first FSAL iteration)
    k1 = f_flat(t0, y_flat)

    # Estimate initial step size if not provided
    if dt0 is None:
        scale = atol + rtol * torch.abs(y_flat)
        # Use abs() for complex support - RMS of absolute values
        d0 = torch.sqrt(torch.mean(torch.abs(y_flat / scale) ** 2))
        d1 = torch.sqrt(torch.mean(torch.abs(k1 / scale) ** 2))
        if d0.real < 1e-5 or d1.real < 1e-5:
            dt0 = 1e-6
        else:
            dt0 = 0.01 * (d0 / d1).real.item()
    dt = dt0

    # Apply dt_max
    if dt_max is not None:
        dt = min(dt, dt_max)

    # Dtype-aware completion tolerance (use real dtype for complex)
    if dtype.is_complex:
        real_dtype = (
            torch.float64 if dtype == torch.complex128 else torch.float32
        )
    else:
        real_dtype = dtype
    t_tol = 100 * torch.finfo(real_dtype).eps * max(abs(t0), abs(t1), 1.0)

    # Storage for interpolant
    t_segments = []
    y_segments = []
    k_segments = []
    segments_dropped = False

    t = t0
    y = y_flat.clone()
    n_steps = 0
    success = (
        None
        if throw
        else torch.ones(batch_shape, dtype=torch.bool, device=device)
    )

    while direction * (t_end - t) > t_tol:
        if n_steps >= max_steps:
            if throw:
                raise MaxStepsExceeded(
                    f"Exceeded maximum number of steps ({max_steps})"
                )
            else:
                y = torch.full_like(y, float("nan"))
                if success is not None:
                    success = torch.zeros_like(success)
                break

        # Clamp step to not overshoot
        dt_step = min(dt, abs(t_end - t))
        h = direction * dt_step

        # Compute RK stages (FSAL: k[0] reuses k1 from previous accepted step)
        k = [None] * 7
        k[0] = k1
        for i in range(1, 7):
            t_i = t + C[i] * h
            y_i = y.clone()
            for j, a_ij in enumerate(A[i]):
                y_i = y_i + h * a_ij * k[j]
            k[i] = f_flat(t_i, y_i)

        # 5th order solution
        y_new = y.clone()
        for i, b in enumerate(B5):
            y_new = y_new + h * b * k[i]

        # 4th order solution for error estimate
        y_err = y.clone()
        for i, b in enumerate(B4):
            y_err = y_err + h * b * k[i]

        # Error estimate with optional seminorm weighting
        error = y_new - y_err
        scale = atol + rtol * torch.maximum(torch.abs(y), torch.abs(y_new))
        if error_weights is not None:
            # Seminorm: scale each component independently by error_weights
            # Higher weight = more sensitive to that component's error
            scale = scale / error_weights
        scaled_error = error / scale

        # Compute error norm (use abs() for complex support)
        if batch_shape:
            state_dims = tuple(range(len(batch_shape), scaled_error.dim()))
            err_norm = torch.sqrt(
                torch.mean(torch.abs(scaled_error) ** 2, dim=state_dims)
            )
            accept_mask = err_norm <= 1.0
            step_accepted = accept_mask.any()
            err_norm_scalar = err_norm.max().item()
        else:
            err_norm_scalar = torch.sqrt(
                torch.mean(torch.abs(scaled_error) ** 2)
            ).item()
            step_accepted = err_norm_scalar <= 1.0
            accept_mask = None

        if step_accepted:
            t_prev = t
            t = t + h
            y_prev = y.clone()

            if accept_mask is not None:
                mask = accept_mask
                for _ in range(y.dim() - len(batch_shape)):
                    mask = mask.unsqueeze(-1)
                y = torch.where(mask, y_new, y)
            else:
                y = y_new

            k1 = k[6]

            if direction > 0:
                t_segments.append((t_prev, t))
                y_segments.append((y_prev, y.clone()))
            else:
                t_segments.append((t, t_prev))
                y_segments.append((y.clone(), y_prev))
            k_segments.append(torch.stack(k))

            if max_segments is not None and len(t_segments) > max_segments:
                t_segments.pop(0)
                y_segments.pop(0)
                k_segments.pop(0)
                if not segments_dropped:
                    warnings.warn(
                        f"max_segments={max_segments} exceeded; dropping oldest segments. "
                        "Interpolation at early times will fail.",
                        UserWarning,
                    )
                    segments_dropped = True

            n_steps += 1

        # Adjust step size
        if err_norm_scalar == 0:
            factor = 2.0
        else:
            factor = 0.9 * (1.0 / err_norm_scalar) ** 0.2
        factor = max(0.1, min(factor, 5.0))
        dt = dt_step * factor

        if dt_max is not None:
            dt = min(dt, dt_max)
        if dt_min is not None and dt < dt_min:
            if throw:
                raise StepSizeError(f"Step size {dt} below minimum {dt_min}")
            else:
                y = torch.full_like(y, float("nan"))
                if success is not None:
                    success = torch.zeros_like(success)
                break

    # Build interpolant
    n_segs = len(t_segments)
    if n_segs == 0:
        t_min, t_max = min(t0, t1), max(t0, t1)
        t_seg_tensor = torch.tensor(
            [[t_min, t_max]], dtype=dtype, device=device
        )
        y_seg_tensor = torch.stack([y_flat, y_flat]).unsqueeze(0)
        k_seg_tensor = torch.stack([k1] * 7).unsqueeze(0)
    else:
        t_seg_tensor = torch.tensor(t_segments, dtype=dtype, device=device)
        y_seg_tensor = torch.stack(
            [torch.stack([s[0], s[1]]) for s in y_segments]
        )
        k_seg_tensor = torch.stack(k_segments)

        if direction < 0:
            sort_indices = t_seg_tensor[:, 0].argsort()
            t_seg_tensor = t_seg_tensor[sort_indices]
            y_seg_tensor = y_seg_tensor[sort_indices]
            k_seg_tensor = k_seg_tensor[sort_indices]

    interp = DP5Interpolant(
        t_seg_tensor, y_seg_tensor, k_seg_tensor, success=success
    )

    # Wrap interpolant for TensorDict
    if is_tensordict:

        class TensorDictInterpolant:
            def __init__(self, base_interp, unflatten_fn):
                self._base = base_interp
                self._unflatten = unflatten_fn
                self.success = base_interp.success
                self.t_points = base_interp.t_points
                self.n_steps = base_interp.n_steps

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

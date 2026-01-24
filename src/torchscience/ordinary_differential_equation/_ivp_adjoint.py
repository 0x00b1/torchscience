"""Adjoint method wrapper for memory-efficient ODE gradients."""

import math
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.ordinary_differential_equation._implicit_adjoint import (
    bdf1_adjoint_step,
)
from torchscience.ordinary_differential_equation._sparse_jacobian import (
    SparseJacobianContext,
)
from torchscience.ordinary_differential_equation._tensordict_utils import (
    flatten_state,
)


class InterpolationAccuracyWarning(UserWarning):
    """Warning when interpolation accuracy may impact adjoint gradients."""

    pass


class AdjointDivergedError(RuntimeError):
    """Raised when adjoint state becomes non-finite."""

    def __init__(self, t: float, norm: float, direction: str = "explosion"):
        super().__init__(
            f"Adjoint diverged at t={t:.4f} (norm={norm:.2e}, {direction}). "
            f"Consider: (1) reducing integration time, (2) using checkpointing, "
            f"(3) enabling gradient_clip in adjoint_options."
        )
        self.t = t
        self.norm = norm
        self.direction = direction


class AdjointStabilityWarning(UserWarning):
    """Warning when adjoint may be unstable."""

    pass


class BacksolveAdjointWarning(UserWarning):
    """Warning when BacksolveAdjoint may be numerically unstable."""

    pass


def _check_adjoint_stability(
    a: torch.Tensor,
    a_prev: torch.Tensor,
    t: float,
    dt: float,
    warned: list,
) -> None:
    """Check adjoint stability and warn/raise as appropriate.

    Parameters
    ----------
    a : Tensor
        Current adjoint state
    a_prev : Tensor
        Previous adjoint state
    t : float
        Current time
    dt : float
        Step size
    warned : list of bool
        Single-element list used as mutable flag to avoid repeated warnings
    """
    a_norm = a.norm().item()
    a_prev_norm = a_prev.norm().item()

    # Check for explosion (non-finite or very large)
    if not torch.isfinite(a).all() or a_norm > 1e30:
        raise AdjointDivergedError(t, a_norm, direction="explosion")

    # Check for rapid growth
    if a_prev_norm > 1e-30 and not warned[0]:
        growth_rate = (
            math.log(a_norm / a_prev_norm) / dt if a_norm > a_prev_norm else 0
        )
        if growth_rate > 1.0:  # e-folding time < 1
            warnings.warn(
                f"Adjoint growing rapidly at t={t:.4f} "
                f"(rate = {growth_rate:.2e}/unit time). "
                f"This may indicate unstable dynamics. "
                f"Consider reducing integration time.",
                AdjointStabilityWarning,
            )
            warned[0] = True

    # Check for vanishing
    if a_norm < 1e-30 and a_prev_norm > 1e-20:
        if not warned[0]:
            warnings.warn(
                f"Adjoint vanishing at t={t:.4f} (norm={a_norm:.2e}). "
                f"Gradients may be numerically zero.",
                AdjointStabilityWarning,
            )
            warned[0] = True


class _InterpolantHolder:
    """Holds interpolant from forward pass for use after apply().

    This is used to pass the interpolant out of the autograd.Function.apply()
    call without requiring a second forward solve. Non-tensor arguments to
    apply() are passed through unchanged, allowing us to populate this holder
    during forward() and retrieve the interpolant afterward.
    """

    def __init__(self):
        self.interp = None


def _kahan_add(sum_val, compensation, addend):
    """Kahan summation for numerically stable accumulation.

    Returns (new_sum, new_compensation)
    """
    y = addend - compensation
    t = sum_val + y
    new_compensation = (t - sum_val) - y
    return t, new_compensation


# TODO: The adjoint function classes (_AdjointODEFunction, _AdaptiveAdjointFunction,
# _CheckpointedAdjointFunction, _BinomialAdjointFunction) share significant code
# duplication in their backward passes (compute_vjp, accumulate_param_grads, RK4/Euler
# integration loops). Consider extracting common adjoint integration logic into a
# shared base class or helper functions in a future refactor.


class _AdjointODEFunction(torch.autograd.Function):
    """
    Custom autograd function implementing the continuous adjoint method.

    Forward pass: Solve the ODE normally
    Backward pass: Solve augmented adjoint ODE backwards in time
    """

    @staticmethod
    def forward(
        ctx,
        y0_flat: torch.Tensor,
        t0: float,
        t1: float,
        solver: Callable,
        f_flat: Callable,
        solver_kwargs: dict,
        adjoint_options: Optional[dict],
        interp_holder: _InterpolantHolder,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: solve ODE and store info for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context for saving tensors
        y0_flat : Tensor
            Flattened initial state
        t0, t1 : float
            Integration interval
        solver : callable
            ODE solver function
        f_flat : callable
            Dynamics function for flattened state
        solver_kwargs : dict
            Additional kwargs for solver
        adjoint_options : dict, optional
            Options for adjoint integration (method, n_steps)
        interp_holder : _InterpolantHolder
            Holder object to store interpolant for retrieval after apply()
        *params : Tensors
            Learnable parameters to compute gradients for

        Returns
        -------
        y_final_flat : Tensor
            Flattened final state
        """
        with torch.no_grad():
            y_final, interp = solver(
                f_flat, y0_flat, t_span=(t0, t1), **solver_kwargs
            )

        # Store in holder for wrapped_solver to retrieve (no second solve needed)
        interp_holder.interp = interp

        # Save for backward
        ctx.save_for_backward(y0_flat, y_final, *params)
        ctx.solver = solver
        ctx.f_flat = f_flat
        ctx.solver_kwargs = solver_kwargs
        ctx.adjoint_options = adjoint_options
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.interp = interp
        ctx.n_params = len(params)

        return y_final.clone()

    @staticmethod
    def backward(ctx, grad_y_final: torch.Tensor) -> Tuple[Any, ...]:
        """
        Backward pass: solve adjoint ODE to compute gradients.

        The adjoint equation is:
            da/dt = -a^T @ (df/dy)

        where a is the adjoint variable (gradient w.r.t. state).

        Parameter gradients are computed as:
            dL/dtheta = integral_t0^t1 a^T @ (df/dtheta) dt
        """
        saved = ctx.saved_tensors
        y0_flat = saved[0]
        y_final = saved[1]
        params = saved[2:]

        solver = ctx.solver
        f_flat = ctx.f_flat
        solver_kwargs = ctx.solver_kwargs
        adjoint_options = ctx.adjoint_options or {}
        t0, t1 = ctx.t0, ctx.t1
        interp = ctx.interp
        n_params = ctx.n_params

        # Check interpolant type and warn if using linear interpolation
        interp_type = type(interp).__name__
        if "Linear" in interp_type or interp_type in [
            "_LinearInterpolant",
            "LinearInterpolant",
        ]:
            warnings.warn(
                f"Using {interp_type} for adjoint integration. "
                f"Linear interpolation has O(dt^2) error which may degrade "
                f"gradient accuracy. Consider using a higher-order solver "
                f"(e.g., dormand_prince_5) for better results.",
                InterpolationAccuracyWarning,
            )

        # Get adjoint integration options
        method = adjoint_options.get("method", "rk4")  # Default to RK4
        n_steps = adjoint_options.get("n_steps", 100)
        mixed_precision = adjoint_options.get("mixed_precision", False)

        # Get gradient clipping options
        gradient_clip = adjoint_options.get("gradient_clip", None)
        gradient_clip_mode = adjoint_options.get(
            "gradient_clip_mode", "norm"
        )  # "norm" or "value"
        clip_warned = [False]

        # Validate gradient_clip_mode
        if gradient_clip is not None and gradient_clip_mode not in (
            "norm",
            "value",
        ):
            raise ValueError(
                f"Invalid gradient_clip_mode: {gradient_clip_mode}. "
                f"Must be 'norm' or 'value'."
            )

        def clip_adjoint(a_val):
            """Clip adjoint state to prevent explosion."""
            if gradient_clip is None:
                return a_val

            if gradient_clip_mode == "norm":
                a_norm = a_val.norm()
                if a_norm > gradient_clip:
                    if not clip_warned[0]:
                        warnings.warn(
                            f"Adjoint clipped: norm {a_norm:.2e} > {gradient_clip:.2e}",
                            AdjointStabilityWarning,
                        )
                        clip_warned[0] = True
                    return a_val * (gradient_clip / a_norm)
            elif gradient_clip_mode == "value":
                clipped = a_val.clamp(-gradient_clip, gradient_clip)
                if not torch.equal(a_val, clipped) and not clip_warned[0]:
                    warnings.warn(
                        f"Adjoint values clipped to [-{gradient_clip}, {gradient_clip}]",
                        AdjointStabilityWarning,
                    )
                    clip_warned[0] = True
                return clipped
            return a_val

        # Check for sparse Jacobian option
        sparsity = adjoint_options.get("sparsity_pattern")
        sparse_ctx = (
            SparseJacobianContext(sparsity) if sparsity is not None else None
        )

        # Initial adjoint state (gradient w.r.t. final state)
        a = grad_y_final.clone()

        # Accumulate parameter gradients with Kahan summation for stability
        # Use float64 for accumulation if mixed_precision is enabled
        accum_dtype = torch.float64 if mixed_precision else None
        if accum_dtype is not None:
            param_grads = [
                torch.zeros_like(p, dtype=accum_dtype) for p in params
            ]
            param_grad_compensations = [
                torch.zeros_like(p, dtype=accum_dtype) for p in params
            ]
        else:
            param_grads = [torch.zeros_like(p) for p in params]
            param_grad_compensations = [torch.zeros_like(p) for p in params]

        # Step size for adjoint integration
        dt = (t1 - t0) / n_steps

        def compute_vjp(t_val, a_val, y_val):
            """Compute a^T @ (df/dy) at (t, y) using VJP."""
            with torch.enable_grad():
                y_local = y_val.clone().requires_grad_(True)
                f_val = f_flat(t_val, y_local)

            if not f_val.requires_grad:
                return torch.zeros_like(a_val)

            # Use sparse VJP if sparsity pattern is provided
            if sparse_ctx is not None:
                return sparse_ctx.vjp(f_val, y_local, a_val)

            vjp = torch.autograd.grad(
                f_val,
                y_local,
                grad_outputs=a_val,
                retain_graph=True,
                allow_unused=True,
            )[0]
            return vjp if vjp is not None else torch.zeros_like(a_val)

        def accumulate_param_grads(t_val, a_val, y_val, weight=1.0):
            """Accumulate parameter gradients: a^T @ (df/dtheta) using Kahan summation."""
            with torch.enable_grad():
                y_local = y_val.clone().requires_grad_(True)
                f_val = f_flat(t_val, y_local)

            if not f_val.requires_grad:
                return

            for i, p in enumerate(params):
                if p.requires_grad:
                    grads = torch.autograd.grad(
                        f_val,
                        p,
                        grad_outputs=a_val,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    if grads[0] is not None:
                        weighted_contribution = weight * dt * grads[0]
                        param_grads[i], param_grad_compensations[i] = (
                            _kahan_add(
                                param_grads[i],
                                param_grad_compensations[i],
                                weighted_contribution,
                            )
                        )

        t = t1

        def clamp_time(t_val):
            """Clamp time value to [t0, t1] to handle floating-point drift."""
            return max(t0, min(t1, t_val))

        # Stability monitoring state
        warned = [False]
        a_prev = a.clone()

        if method == "rk4":
            # RK4 integration for adjoint ODE backwards from t1 to t0
            # The adjoint ODE is: da/dt = -a^T @ (df/dy)
            # When integrating backwards (t decreasing), we use negative time step:
            # a(t-dt) = a(t) - dt * [da/dt] = a(t) - dt * [-a^T @ df/dy] = a(t) + dt * [a^T @ df/dy]
            # So the RHS for backward integration is: g(t, a) = a^T @ (df/dy) = compute_vjp
            for _ in range(n_steps):
                t_mid = clamp_time(t - dt / 2)
                t_next = clamp_time(t - dt)

                # Get y values at different time points from interpolant
                y_t = interp(clamp_time(t))
                y_mid = interp(t_mid)
                y_next = interp(t_next)

                # RK4 stages for backward integration
                # g(t, a) = compute_vjp(t, a, y) is the RHS after sign flip for backward stepping
                k1 = compute_vjp(t, a, y_t)

                a_k2 = a + (dt / 2) * k1
                k2 = compute_vjp(t_mid, a_k2, y_mid)

                a_k3 = a + (dt / 2) * k2
                k3 = compute_vjp(t_mid, a_k3, y_mid)

                a_k4 = a + dt * k3
                k4 = compute_vjp(t_next, a_k4, y_next)

                # Parameter gradient accumulation using Simpson's rule with RK4 stage values
                # Simpson's rule integrates parameter gradients with weights (1, 4, 1)/6
                # using the adjoint values from RK4 stages k1, k2, k3, k4 for accuracy
                accumulate_param_grads(t, a, y_t, weight=1.0 / 6)
                # For midpoint, use average of k2 and k3 adjoint values
                a_mid = (a_k2 + a_k3) / 2
                accumulate_param_grads(t_mid, a_mid, y_mid, weight=4.0 / 6)
                accumulate_param_grads(t_next, a_k4, y_next, weight=1.0 / 6)

                # RK4 update
                a = a + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

                # Apply gradient clipping
                a = clip_adjoint(a)

                # Check adjoint stability
                _check_adjoint_stability(a, a_prev, t, dt, warned)
                a_prev = a.clone()

                t = t_next

        elif method == "euler":
            # Forward Euler for adjoint ODE backwards integration
            # The adjoint ODE is: da/dt = -a^T @ (df/dy)
            # When integrating backwards (t decreasing), we use: a(t-dt) = a(t) + dt * [a^T @ df/dy]
            for _ in range(n_steps):
                t_prev = clamp_time(t - dt)
                y_at_t = interp(clamp_time(t))

                # Adjoint update: da/dt = -a^T @ (df/dy)
                # Approximate: a_prev = a + dt * a^T @ (df/dy)
                vjp_result = compute_vjp(t, a, y_at_t)
                a = a + dt * vjp_result

                # Apply gradient clipping
                a = clip_adjoint(a)

                # Check adjoint stability
                _check_adjoint_stability(a, a_prev, t, dt, warned)
                a_prev = a.clone()

                # Accumulate parameter gradients
                accumulate_param_grads(t, a, y_at_t, weight=1.0)

                t = t_prev

        else:
            raise ValueError(
                f"Unknown adjoint method: {method}. Use 'rk4' or 'euler'."
            )

        # Gradient w.r.t. y0 is the final adjoint state
        grad_y0 = a

        # Convert parameter gradients back to original dtype if mixed precision was used
        if mixed_precision:
            param_grads = [g.to(p.dtype) for g, p in zip(param_grads, params)]

        # Return gradients in same order as forward inputs
        # (y0_flat, t0, t1, solver, f_flat, solver_kwargs, adjoint_options, interp_holder, *params)
        return (grad_y0, None, None, None, None, None, None, None) + tuple(
            param_grads
        )


class _AdaptiveAdjointFunction(torch.autograd.Function):
    """
    Adaptive adjoint method using the same solver for backward integration.

    Instead of using fixed-step RK4/Euler for the adjoint ODE, this uses the
    same adaptive solver (e.g., DP5) for the backward integration. This
    automatically matches forward and backward accuracy.

    The augmented state is: [adjoint (n_state), param_grads (n_param_total)]
    where n_param_total = sum of numel() for each parameter.
    """

    @staticmethod
    def forward(
        ctx,
        y0_flat: torch.Tensor,
        t0: float,
        t1: float,
        solver: Callable,
        f_flat: Callable,
        solver_kwargs: dict,
        adjoint_options: Optional[dict],
        interp_holder: _InterpolantHolder,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: solve ODE and store info for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context for saving tensors
        y0_flat : Tensor
            Flattened initial state
        t0, t1 : float
            Integration interval
        solver : callable
            ODE solver function
        f_flat : callable
            Dynamics function for flattened state
        solver_kwargs : dict
            Additional kwargs for solver
        adjoint_options : dict, optional
            Options for adjoint integration
        interp_holder : _InterpolantHolder
            Holder object to store interpolant for retrieval after apply()
        *params : Tensors
            Learnable parameters to compute gradients for

        Returns
        -------
        y_final_flat : Tensor
            Flattened final state
        """
        with torch.no_grad():
            y_final, interp = solver(
                f_flat, y0_flat, t_span=(t0, t1), **solver_kwargs
            )

        # Store in holder for wrapped_solver to retrieve (no second solve needed)
        interp_holder.interp = interp

        # Save for backward
        ctx.save_for_backward(y0_flat, y_final, *params)
        ctx.solver = solver
        ctx.f_flat = f_flat
        ctx.solver_kwargs = solver_kwargs
        ctx.adjoint_options = adjoint_options
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.interp = interp
        ctx.n_params = len(params)

        return y_final.clone()

    @staticmethod
    def backward(ctx, grad_y_final: torch.Tensor) -> Tuple[Any, ...]:
        """
        Backward pass: solve adjoint ODE using the same adaptive solver.

        The adjoint equation is:
            da/dt = -a^T @ (df/dy)

        where a is the adjoint variable (gradient w.r.t. state).

        Parameter gradients are computed by augmenting the state to include
        them and integrating:
            d(param_grad)/dt = a^T @ (df/dparam)

        The augmented state [a, param_grads] is solved backwards using the
        same adaptive solver used for the forward pass.
        """
        saved = ctx.saved_tensors
        # saved[0] is y0_flat, saved[1] is y_final (not needed in backward)
        params = saved[2:]

        solver = ctx.solver
        f_flat = ctx.f_flat
        solver_kwargs = ctx.solver_kwargs
        adjoint_options = ctx.adjoint_options or {}
        t0, t1 = ctx.t0, ctx.t1
        interp = ctx.interp

        # Compute total number of parameter elements
        param_shapes = [p.shape for p in params]
        param_numels = [p.numel() for p in params]
        n_param_total = sum(param_numels)

        # State dimensions
        n_state = grad_y_final.numel()

        # Initial augmented state at t1: [adjoint, param_grads]
        # adjoint = grad_y_final (gradient w.r.t. final state)
        # param_grads = 0 (accumulated from t1 to t0)
        a0 = grad_y_final.flatten()
        param_grad0 = torch.zeros(
            n_param_total, dtype=grad_y_final.dtype, device=grad_y_final.device
        )
        aug_state0 = torch.cat([a0, param_grad0])

        def augmented_dynamics(t_backward, aug_state):
            """
            Augmented dynamics for adjoint + parameter gradients.

            The backward time t_backward runs from 0 to (t1 - t0).
            The actual forward time is: t_forward = t1 - t_backward.

            State layout: [adjoint (n_state), param_grads (n_param_total)]

            Dynamics:
                d(adjoint)/dt_backward = a^T @ (df/dy)  [sign flipped for backward]
                d(param_grads)/dt_backward = a^T @ (df/dparam)
            """
            # Extract adjoint from augmented state
            a = aug_state[:n_state]

            # Convert backward time to forward time
            t_forward = t1 - t_backward

            # Get y at this forward time from interpolant
            y_at_t = interp(t_forward)

            # Compute VJPs
            with torch.enable_grad():
                y_local = y_at_t.clone().requires_grad_(True)
                f_val = f_flat(t_forward, y_local)

            # da/dt_backward = a^T @ (df/dy)
            # (This is the RHS with sign already flipped for backward integration)
            if f_val.requires_grad:
                vjp_y = torch.autograd.grad(
                    f_val,
                    y_local,
                    grad_outputs=a.view_as(f_val),
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                da = (
                    vjp_y.flatten()
                    if vjp_y is not None
                    else torch.zeros_like(a)
                )
            else:
                da = torch.zeros_like(a)

            # d(param_grads)/dt_backward = a^T @ (df/dparam)
            dparam_grads = []
            for p in params:
                if p.requires_grad and f_val.requires_grad:
                    vjp_p = torch.autograd.grad(
                        f_val,
                        p,
                        grad_outputs=a.view_as(f_val),
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if vjp_p is not None:
                        dparam_grads.append(vjp_p.flatten())
                    else:
                        dparam_grads.append(
                            torch.zeros(
                                p.numel(), dtype=a.dtype, device=a.device
                            )
                        )
                else:
                    dparam_grads.append(
                        torch.zeros(p.numel(), dtype=a.dtype, device=a.device)
                    )

            if dparam_grads:
                dparam = torch.cat(dparam_grads)
            else:
                dparam = torch.zeros(
                    n_param_total, dtype=a.dtype, device=a.device
                )

            return torch.cat([da, dparam])

        # Solve augmented adjoint ODE backward from t1 to t0
        # We transform to forward time: t_backward in [0, t1 - t0]
        t_span_backward = (0.0, t1 - t0)

        # Use the same solver for backward integration
        # Filter solver_kwargs to remove any that might cause issues
        backward_kwargs = {
            k: v
            for k, v in solver_kwargs.items()
            if k
            not in [
                "max_segments"
            ]  # max_segments can cause issues for short integrations
        }

        with torch.no_grad():
            aug_final, _ = solver(
                augmented_dynamics,
                aug_state0,
                t_span=t_span_backward,
                **backward_kwargs,
            )

        # Extract grad_y0 and param_grads from final augmented state
        grad_y0 = aug_final[:n_state].view_as(grad_y_final)

        # Extract and reshape parameter gradients
        param_grads = []
        offset = n_state
        for shape, numel in zip(param_shapes, param_numels):
            param_grad = aug_final[offset : offset + numel].view(shape)
            param_grads.append(param_grad)
            offset += numel

        # Return gradients in same order as forward inputs
        # (y0_flat, t0, t1, solver, f_flat, solver_kwargs, adjoint_options, interp_holder, *params)
        return (grad_y0, None, None, None, None, None, None, None) + tuple(
            param_grads
        )


class _CheckpointedAdjointFunction(torch.autograd.Function):
    """
    Checkpointed adjoint using linear checkpointing.

    Instead of storing full trajectory, stores N checkpoints and
    recomputes between checkpoints during backward.

    This trades compute for memory: with N checkpoints over T time units,
    memory is O(N) instead of O(n_steps), but backward requires O(n_steps/N)
    extra forward solves per checkpoint.
    """

    @staticmethod
    def forward(
        ctx,
        y0_flat: torch.Tensor,
        t0: float,
        t1: float,
        solver: Callable,
        f_flat: Callable,
        solver_kwargs: dict,
        adjoint_options: Optional[dict],
        n_checkpoints: int,
        interp_holder: _InterpolantHolder,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: solve ODE and store checkpoint states.

        Parameters
        ----------
        ctx : Context
            Autograd context for saving tensors
        y0_flat : Tensor
            Flattened initial state
        t0, t1 : float
            Integration interval
        solver : callable
            ODE solver function
        f_flat : callable
            Dynamics function for flattened state
        solver_kwargs : dict
            Additional kwargs for solver
        adjoint_options : dict, optional
            Options for adjoint integration
        n_checkpoints : int
            Number of checkpoints (N+1 states will be stored)
        interp_holder : _InterpolantHolder
            Holder object to store interpolant for retrieval after apply()
        *params : Tensors
            Learnable parameters to compute gradients for

        Returns
        -------
        y_final_flat : Tensor
            Flattened final state
        """
        # Compute checkpoint times (N+1 points including start and end)
        checkpoint_times = torch.linspace(t0, t1, n_checkpoints + 1)

        # Run forward solve to get full solution
        with torch.no_grad():
            y_final, interp = solver(
                f_flat, y0_flat, t_span=(t0, t1), **solver_kwargs
            )

        # Store interpolant in holder for wrapped_solver to retrieve
        interp_holder.interp = interp

        # Store checkpoints (states at checkpoint times)
        checkpoints = []
        for t_ckpt in checkpoint_times:
            y_ckpt = interp(t_ckpt.item())
            checkpoints.append(y_ckpt.clone())

        # Save checkpoints and params
        ctx.save_for_backward(*checkpoints, *params)
        ctx.checkpoint_times = checkpoint_times
        ctx.n_checkpoints = n_checkpoints
        ctx.solver = solver
        ctx.f_flat = f_flat
        ctx.solver_kwargs = solver_kwargs
        ctx.adjoint_options = adjoint_options
        ctx.n_params = len(params)
        ctx.t0 = t0
        ctx.t1 = t1

        return y_final.clone()

    @staticmethod
    def backward(ctx, grad_y_final: torch.Tensor) -> Tuple[Any, ...]:
        """
        Backward pass: solve adjoint ODE using checkpointed recomputation.

        For each segment (from checkpoint i to checkpoint i+1):
        1. Recompute forward trajectory using solver
        2. Solve adjoint ODE on this segment using the recomputed trajectory
        3. Accumulate parameter gradients
        """
        n_checkpoints = ctx.n_checkpoints
        n_params = ctx.n_params
        saved = ctx.saved_tensors
        checkpoints = list(saved[: n_checkpoints + 1])
        params = saved[n_checkpoints + 1 :]

        checkpoint_times = ctx.checkpoint_times
        solver = ctx.solver
        f_flat = ctx.f_flat
        solver_kwargs = ctx.solver_kwargs
        adjoint_options = ctx.adjoint_options or {}
        t0, t1 = ctx.t0, ctx.t1

        method = adjoint_options.get("method", "rk4")
        total_n_steps = adjoint_options.get("n_steps", 100)
        n_steps_per_segment = max(1, total_n_steps // n_checkpoints)

        # Initialize adjoint at final time
        a = grad_y_final.clone()
        param_grads = [torch.zeros_like(p) for p in params]
        param_grad_compensations = [torch.zeros_like(p) for p in params]

        def compute_vjp(t_val, a_val, y_val):
            """Compute a^T @ (df/dy) at (t, y) using VJP."""
            with torch.enable_grad():
                y_local = y_val.clone().requires_grad_(True)
                f_val = f_flat(t_val, y_local)

            if not f_val.requires_grad:
                return torch.zeros_like(a_val)

            vjp = torch.autograd.grad(
                f_val,
                y_local,
                grad_outputs=a_val,
                retain_graph=True,
                allow_unused=True,
            )[0]
            return vjp if vjp is not None else torch.zeros_like(a_val)

        def accumulate_param_grads(t_val, a_val, y_val, dt_seg, weight=1.0):
            """Accumulate parameter gradients: a^T @ (df/dtheta) using Kahan summation."""
            with torch.enable_grad():
                y_local = y_val.clone().requires_grad_(True)
                f_val = f_flat(t_val, y_local)

            if not f_val.requires_grad:
                return

            for i, p in enumerate(params):
                if p.requires_grad:
                    grads = torch.autograd.grad(
                        f_val,
                        p,
                        grad_outputs=a_val,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    if grads[0] is not None:
                        weighted_contribution = weight * dt_seg * grads[0]
                        param_grads[i], param_grad_compensations[i] = (
                            _kahan_add(
                                param_grads[i],
                                param_grad_compensations[i],
                                weighted_contribution,
                            )
                        )

        # Process each segment backwards (from last checkpoint to first)
        for seg_idx in range(n_checkpoints - 1, -1, -1):
            t_start = checkpoint_times[seg_idx].item()
            t_end = checkpoint_times[seg_idx + 1].item()
            y_start = checkpoints[seg_idx]

            # Recompute trajectory for this segment
            with torch.no_grad():
                _, seg_interp = solver(
                    f_flat, y_start, t_span=(t_start, t_end), **solver_kwargs
                )

            # Step size for adjoint integration on this segment
            dt = (t_end - t_start) / n_steps_per_segment
            t = t_end

            def clamp_time(t_val):
                """Clamp time value to [t_start, t_end] for this segment."""
                return max(t_start, min(t_end, t_val))

            if method == "rk4":
                for _ in range(n_steps_per_segment):
                    t_mid = clamp_time(t - dt / 2)
                    t_next = clamp_time(t - dt)

                    # Get y values at different time points from segment interpolant
                    y_t = seg_interp(clamp_time(t))
                    y_mid = seg_interp(t_mid)
                    y_next = seg_interp(t_next)

                    # RK4 stages for backward integration
                    k1 = compute_vjp(t, a, y_t)

                    a_k2 = a + (dt / 2) * k1
                    k2 = compute_vjp(t_mid, a_k2, y_mid)

                    a_k3 = a + (dt / 2) * k2
                    k3 = compute_vjp(t_mid, a_k3, y_mid)

                    a_k4 = a + dt * k3
                    k4 = compute_vjp(t_next, a_k4, y_next)

                    # Parameter gradient accumulation using Simpson's rule
                    accumulate_param_grads(t, a, y_t, dt, weight=1.0 / 6)
                    a_mid = (a_k2 + a_k3) / 2
                    accumulate_param_grads(
                        t_mid, a_mid, y_mid, dt, weight=4.0 / 6
                    )
                    accumulate_param_grads(
                        t_next, a_k4, y_next, dt, weight=1.0 / 6
                    )

                    # RK4 update
                    a = a + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

                    t = t_next

            elif method == "euler":
                for _ in range(n_steps_per_segment):
                    t_prev = clamp_time(t - dt)
                    y_at_t = seg_interp(clamp_time(t))

                    vjp_result = compute_vjp(t, a, y_at_t)
                    a = a + dt * vjp_result

                    accumulate_param_grads(t, a, y_at_t, dt, weight=1.0)

                    t = t_prev

            else:
                raise ValueError(
                    f"Unknown adjoint method: {method}. Use 'rk4' or 'euler'."
                )

        # Gradient w.r.t. y0 is the final adjoint state
        grad_y0 = a

        # Return gradients in same order as forward inputs
        # (y0_flat, t0, t1, solver, f_flat, solver_kwargs, adjoint_options, n_checkpoints, interp_holder, *params)
        return (
            grad_y0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ) + tuple(param_grads)


class _BinomialAdjointFunction(torch.autograd.Function):
    """
    Binomial checkpointing adjoint using the Revolve algorithm.

    This achieves O(log n) memory usage with O(n log n) recomputation
    by placing checkpoints at optimal positions determined by binomial
    coefficients.

    The key difference from linear checkpointing:
    - Linear: N checkpoints equally spaced -> O(N) memory, O(n^2/N) recompute
    - Binomial: O(log n) checkpoints optimally placed -> O(log n) memory, O(n log n) recompute

    The implementation stores checkpoints during the forward pass at times
    determined by the binomial schedule, then during backward processes
    segments in reverse order, recomputing each segment from its starting
    checkpoint.
    """

    @staticmethod
    def forward(
        ctx,
        y0_flat: torch.Tensor,
        t0: float,
        t1: float,
        solver: Callable,
        f_flat: Callable,
        solver_kwargs: dict,
        adjoint_options: Optional[dict],
        interp_holder: _InterpolantHolder,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: solve ODE and store O(log n) checkpoint states.

        Parameters
        ----------
        ctx : Context
            Autograd context for saving tensors
        y0_flat : Tensor
            Flattened initial state
        t0, t1 : float
            Integration interval
        solver : callable
            ODE solver function
        f_flat : callable
            Dynamics function for flattened state
        solver_kwargs : dict
            Additional kwargs for solver
        adjoint_options : dict, optional
            Options for adjoint integration
        interp_holder : _InterpolantHolder
            Holder object to store interpolant for retrieval after apply()
        *params : Tensors
            Learnable parameters to compute gradients for

        Returns
        -------
        y_final_flat : Tensor
            Flattened final state
        """
        from torchscience.ordinary_differential_equation._checkpointing import (
            BinomialCheckpointSchedule,
        )

        # Run forward solve to get full solution
        with torch.no_grad():
            y_final, interp = solver(
                f_flat, y0_flat, t_span=(t0, t1), **solver_kwargs
            )

        # Store interpolant in holder for wrapped_solver to retrieve
        interp_holder.interp = interp

        # Determine number of segments based on estimated steps or time span.
        # Use O(log n) checkpoints where n is roughly the number of steps.
        # We estimate n from the interpolant's stored points if available.
        n_points = getattr(interp, "n_steps", None)
        if n_points is None:
            # Heuristic: When the interpolant doesn't expose step count, estimate
            # based on time span. The factor of 10 steps per unit time is a rough
            # approximation for typical stiff/non-stiff ODEs. The minimum of 10
            # ensures we always have at least a few checkpoints even for short
            # integrations. This affects only checkpoint placement, not accuracy.
            n_points = max(10, int((t1 - t0) * 10))

        schedule = BinomialCheckpointSchedule.from_n_steps(n_points, t0, t1)

        # Store checkpoints at schedule times
        checkpoints = []
        for t_ckpt in schedule.checkpoint_times:
            y_ckpt = interp(t_ckpt)
            checkpoints.append(y_ckpt.clone())

        # Save checkpoints and params
        ctx.save_for_backward(*checkpoints, *params)
        ctx.checkpoint_times = schedule.checkpoint_times
        ctx.n_segments = schedule.n_segments
        ctx.solver = solver
        ctx.f_flat = f_flat
        ctx.solver_kwargs = solver_kwargs
        ctx.adjoint_options = adjoint_options
        ctx.n_params = len(params)
        ctx.t0 = t0
        ctx.t1 = t1

        return y_final.clone()

    @staticmethod
    def backward(ctx, grad_y_final: torch.Tensor) -> Tuple[Any, ...]:
        """
        Backward pass: solve adjoint ODE using binomial checkpointed recomputation.

        For each segment (from checkpoint i to checkpoint i+1):
        1. Recompute forward trajectory using solver from checkpoint
        2. Solve adjoint ODE on this segment using the recomputed trajectory
        3. Accumulate parameter gradients

        The key insight is that with O(log n) checkpoints, each segment is
        O(n / log n) steps, so total recomputation is O(n) per segment,
        giving O(n log n) total recomputation.
        """
        n_segments = ctx.n_segments
        n_params = ctx.n_params
        saved = ctx.saved_tensors
        n_checkpoints = n_segments + 1  # Checkpoints at segment boundaries
        checkpoints = list(saved[:n_checkpoints])
        params = saved[n_checkpoints:]

        checkpoint_times = ctx.checkpoint_times
        solver = ctx.solver
        f_flat = ctx.f_flat
        solver_kwargs = ctx.solver_kwargs
        adjoint_options = ctx.adjoint_options or {}
        t0, t1 = ctx.t0, ctx.t1

        # Get adjoint integration method (for backward pass, not the binomial selection)
        # If method is "binomial", use "rk4" as default for the actual integration
        adj_method = adjoint_options.get("method", "rk4")
        if adj_method == "binomial":
            adj_method = "rk4"
        method = adjoint_options.get("adjoint_method", adj_method)
        total_n_steps = adjoint_options.get("n_steps", 100)
        n_steps_per_segment = max(1, total_n_steps // n_segments)

        # Initialize adjoint at final time
        a = grad_y_final.clone()
        param_grads = [torch.zeros_like(p) for p in params]
        param_grad_compensations = [torch.zeros_like(p) for p in params]

        def compute_vjp(t_val, a_val, y_val):
            """Compute a^T @ (df/dy) at (t, y) using VJP."""
            with torch.enable_grad():
                y_local = y_val.clone().requires_grad_(True)
                f_val = f_flat(t_val, y_local)

            if not f_val.requires_grad:
                return torch.zeros_like(a_val)

            vjp = torch.autograd.grad(
                f_val,
                y_local,
                grad_outputs=a_val,
                retain_graph=True,
                allow_unused=True,
            )[0]
            return vjp if vjp is not None else torch.zeros_like(a_val)

        def accumulate_param_grads(t_val, a_val, y_val, dt_seg, weight=1.0):
            """Accumulate parameter gradients: a^T @ (df/dtheta) using Kahan summation."""
            with torch.enable_grad():
                y_local = y_val.clone().requires_grad_(True)
                f_val = f_flat(t_val, y_local)

            if not f_val.requires_grad:
                return

            for i, p in enumerate(params):
                if p.requires_grad:
                    grads = torch.autograd.grad(
                        f_val,
                        p,
                        grad_outputs=a_val,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    if grads[0] is not None:
                        weighted_contribution = weight * dt_seg * grads[0]
                        param_grads[i], param_grad_compensations[i] = (
                            _kahan_add(
                                param_grads[i],
                                param_grad_compensations[i],
                                weighted_contribution,
                            )
                        )

        # Process each segment backwards (from last checkpoint to first)
        for seg_idx in range(n_segments - 1, -1, -1):
            t_start = checkpoint_times[seg_idx]
            t_end = checkpoint_times[seg_idx + 1]
            y_start = checkpoints[seg_idx]

            # Recompute trajectory for this segment
            with torch.no_grad():
                _, seg_interp = solver(
                    f_flat, y_start, t_span=(t_start, t_end), **solver_kwargs
                )

            # Step size for adjoint integration on this segment
            dt = (t_end - t_start) / n_steps_per_segment
            t = t_end

            def clamp_time(t_val):
                """Clamp time value to [t_start, t_end] for this segment."""
                return max(t_start, min(t_end, t_val))

            if method == "rk4":
                for _ in range(n_steps_per_segment):
                    t_mid = clamp_time(t - dt / 2)
                    t_next = clamp_time(t - dt)

                    # Get y values at different time points from segment interpolant
                    y_t = seg_interp(clamp_time(t))
                    y_mid = seg_interp(t_mid)
                    y_next = seg_interp(t_next)

                    # RK4 stages for backward integration
                    k1 = compute_vjp(t, a, y_t)

                    a_k2 = a + (dt / 2) * k1
                    k2 = compute_vjp(t_mid, a_k2, y_mid)

                    a_k3 = a + (dt / 2) * k2
                    k3 = compute_vjp(t_mid, a_k3, y_mid)

                    a_k4 = a + dt * k3
                    k4 = compute_vjp(t_next, a_k4, y_next)

                    # Parameter gradient accumulation using Simpson's rule
                    accumulate_param_grads(t, a, y_t, dt, weight=1.0 / 6)
                    a_mid = (a_k2 + a_k3) / 2
                    accumulate_param_grads(
                        t_mid, a_mid, y_mid, dt, weight=4.0 / 6
                    )
                    accumulate_param_grads(
                        t_next, a_k4, y_next, dt, weight=1.0 / 6
                    )

                    # RK4 update
                    a = a + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

                    t = t_next

            elif method == "euler":
                for _ in range(n_steps_per_segment):
                    t_prev = clamp_time(t - dt)
                    y_at_t = seg_interp(clamp_time(t))

                    vjp_result = compute_vjp(t, a, y_at_t)
                    a = a + dt * vjp_result

                    accumulate_param_grads(t, a, y_at_t, dt, weight=1.0)

                    t = t_prev

            else:
                raise ValueError(
                    f"Unknown adjoint method: {method}. Use 'rk4' or 'euler'."
                )

        # Gradient w.r.t. y0 is the final adjoint state
        grad_y0 = a

        # Return gradients in same order as forward inputs
        # (y0_flat, t0, t1, solver, f_flat, solver_kwargs, adjoint_options, interp_holder, *params)
        return (grad_y0, None, None, None, None, None, None, None) + tuple(
            param_grads
        )


class _BacksolveAdjointFunction(torch.autograd.Function):
    """
    BacksolveAdjoint: Recompute forward trajectory during backward pass.

    This achieves true O(1) memory by not storing the forward trajectory.
    Instead, it re-integrates the forward ODE backwards (from t1 to t0)
    alongside the adjoint ODE.

    WARNING: Only stable for non-chaotic systems. For chaotic dynamics,
    the backward-integrated trajectory will diverge from the forward one.
    """

    @staticmethod
    def forward(
        ctx,
        y0_flat: torch.Tensor,
        t0: float,
        t1: float,
        solver: Callable,
        f_flat: Callable,
        solver_kwargs: dict,
        adjoint_options: Optional[dict],
        interp_holder: _InterpolantHolder,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: solve ODE and store minimal info for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context for saving tensors
        y0_flat : Tensor
            Flattened initial state
        t0, t1 : float
            Integration interval
        solver : callable
            ODE solver function
        f_flat : callable
            Dynamics function for flattened state
        solver_kwargs : dict
            Additional kwargs for solver
        adjoint_options : dict, optional
            Options for adjoint integration
        interp_holder : _InterpolantHolder
            Holder object to store interpolant for retrieval after apply()
        *params : Tensors
            Learnable parameters to compute gradients for

        Returns
        -------
        y_final_flat : Tensor
            Flattened final state
        """
        with torch.no_grad():
            y_final, interp = solver(
                f_flat, y0_flat, t_span=(t0, t1), **solver_kwargs
            )

        interp_holder.interp = interp

        # Only save minimal data - will recompute trajectory in backward
        ctx.save_for_backward(y0_flat, y_final, *params)
        ctx.solver = solver
        ctx.f_flat = f_flat
        ctx.solver_kwargs = solver_kwargs
        ctx.adjoint_options = adjoint_options
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.n_params = len(params)

        return y_final.clone()

    @staticmethod
    def backward(ctx, grad_y_final: torch.Tensor) -> Tuple[Any, ...]:
        """
        Backward pass: solve augmented ODE to compute gradients.

        The augmented state includes:
        - y: forward state (integrated backwards to verify reconstruction)
        - a: adjoint state (gradient w.r.t. state)
        - g: accumulated parameter gradients

        This integrates all three simultaneously from t1 to t0.
        """
        saved = ctx.saved_tensors
        y0_flat = saved[0]
        y_final = saved[1]
        params = saved[2:]

        solver = ctx.solver
        f_flat = ctx.f_flat
        solver_kwargs = ctx.solver_kwargs.copy()
        adjoint_options = ctx.adjoint_options or {}
        t0, t1 = ctx.t0, ctx.t1

        # Augmented state: [y, adjoint, param_grads]
        n_state = y0_flat.numel()
        n_param_total = sum(p.numel() for p in params)

        def augmented_dynamics(t_backward, aug_state):
            """
            Augmented dynamics for simultaneous forward and adjoint integration.

            The backward time t_backward runs from 0 to (t1 - t0).
            The actual forward time is: t_forward = t1 - t_backward.

            State layout: [y (n_state), a (n_state), g_theta (n_param_total)]

            We integrate in forward t_backward direction, so:
            - dy/d(t_backward) = -f(t_forward, y)  (since dt_forward = -dt_backward)
            - da/d(t_backward) = a^T @ (df/dy)  (sign flipped for backward)
            - dg/d(t_backward) = a^T @ (df/dtheta)
            """
            y = aug_state[:n_state]
            a = aug_state[n_state : 2 * n_state]

            # Convert backward time to forward time
            t_forward = t1 - t_backward

            # Compute f and its derivatives
            with torch.enable_grad():
                y_local = y.clone().requires_grad_(True)
                f_val = f_flat(t_forward, y_local)

            # dy/d(t_backward) = -f (reversing time direction)
            dy_dt = -f_val.flatten()

            # Adjoint: da/d(t_backward) = a^T @ J
            # (sign already flipped for backward time)
            da_dt = torch.zeros_like(a)
            if f_val.requires_grad:
                vjp = torch.autograd.grad(
                    f_val,
                    y_local,
                    grad_outputs=a.reshape(f_val.shape),
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if vjp is not None:
                    da_dt = vjp.flatten()

            # Parameter gradients: dg/d(t_backward) = a^T @ (df/dtheta)
            dparam_dt = []
            for p in params:
                if p.requires_grad:
                    vjp_p = torch.autograd.grad(
                        f_val,
                        p,
                        grad_outputs=a.reshape(f_val.shape),
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if vjp_p is not None:
                        dparam_dt.append(vjp_p.flatten())
                    else:
                        dparam_dt.append(
                            torch.zeros(
                                p.numel(), device=p.device, dtype=p.dtype
                            )
                        )
                else:
                    dparam_dt.append(
                        torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
                    )

            if dparam_dt:
                dparam_dt = torch.cat(dparam_dt)
            else:
                dparam_dt = torch.tensor([], device=y.device, dtype=y.dtype)

            return torch.cat([dy_dt, da_dt, dparam_dt])

        # Initial augmented state at t1 (t_backward = 0)
        aug_y0 = torch.cat(
            [
                y_final.flatten(),
                grad_y_final.flatten(),
                torch.zeros(
                    n_param_total, device=y_final.device, dtype=y_final.dtype
                ),
            ]
        )

        # Solve augmented system in backward time from 0 to (t1 - t0)
        t_span_backward = (0.0, t1 - t0)
        with torch.no_grad():
            aug_final, _ = solver(
                augmented_dynamics,
                aug_y0,
                t_span=t_span_backward,
                **solver_kwargs,
            )

        # Extract results
        y_reconstructed = aug_final[:n_state]
        grad_y0 = aug_final[n_state : 2 * n_state].reshape(y0_flat.shape)

        # Check reconstruction accuracy
        reconstruction_error = (y_reconstructed - y0_flat.flatten()).norm() / (
            y0_flat.norm() + 1e-10
        )
        if reconstruction_error > 0.1:
            warnings.warn(
                f"BacksolveAdjoint reconstruction error: {reconstruction_error:.2e}. "
                f"This may indicate chaotic or unstable dynamics. "
                f"Consider using 'adjoint' or 'checkpoint' methods instead.",
                BacksolveAdjointWarning,
            )

        # Extract parameter gradients
        param_grads = []
        offset = 2 * n_state
        for p in params:
            n_p = p.numel()
            param_grads.append(
                aug_final[offset : offset + n_p].reshape(p.shape)
            )
            offset += n_p

        return (grad_y0, None, None, None, None, None, None, None) + tuple(
            param_grads
        )


class _ImplicitAdjointFunction(torch.autograd.Function):
    """Implicit adjoint using BDF methods for stiff systems."""

    @staticmethod
    def forward(
        ctx,
        y0_flat,
        t0,
        t1,
        solver,
        f_flat,
        solver_kwargs,
        adjoint_options,
        interp_holder,
        *params,
    ):
        with torch.no_grad():
            y_final, interp = solver(
                f_flat, y0_flat, t_span=(t0, t1), **solver_kwargs
            )

        interp_holder.interp = interp

        ctx.save_for_backward(y0_flat, y_final, *params)
        ctx.solver = solver
        ctx.f_flat = f_flat
        ctx.solver_kwargs = solver_kwargs
        ctx.adjoint_options = adjoint_options
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.interp = interp
        ctx.n_params = len(params)

        return y_final.clone()

    @staticmethod
    def backward(ctx, grad_y_final):
        saved = ctx.saved_tensors
        params = saved[2:]

        f_flat = ctx.f_flat
        adjoint_options = ctx.adjoint_options or {}
        t0, t1 = ctx.t0, ctx.t1
        interp = ctx.interp

        n_steps = adjoint_options.get("n_steps", 100)
        dt = (t1 - t0) / n_steps

        # Initialize
        a = grad_y_final.clone()
        param_grads = [torch.zeros_like(p) for p in params]

        t = t1
        for _ in range(n_steps):
            a, step_grads = bdf1_adjoint_step(
                a, t, dt, f_flat, interp, list(params)
            )

            for i, g in enumerate(step_grads):
                param_grads[i] = param_grads[i] + g

            t = t - dt

        grad_y0 = a
        return (grad_y0, None, None, None, None, None, None, None) + tuple(
            param_grads
        )


def adjoint(
    solver: Callable,
    checkpoints: Optional[int] = None,
    params: Optional[List[torch.Tensor]] = None,
    adjoint_options: Optional[dict] = None,
) -> Callable:
    """
    Wrap a solver to use adjoint method for gradients.

    The adjoint method computes gradients by solving an augmented ODE
    backwards in time, using O(1) memory for the autograd graph instead
    of O(n_steps).

    Parameters
    ----------
    solver : callable
        Any ODE solver function (euler, dormand_prince_5, etc.)
    checkpoints : int, optional
        Number of checkpoints for memory/compute tradeoff.
        None = automatic selection (currently unused, reserved for future).
    params : list, optional
        List of tensors requiring gradients. If None (default), parameters
        are extracted from the dynamics function's closure (may be unreliable).
        If an empty list [], no parameter gradients are computed (forward-only).
        If a list of tensors, gradients are computed for exactly those tensors.
    adjoint_options : dict, optional
        Options for adjoint integration. Supported keys:
        - 'method': str, integration method. Options:
            - 'rk4': Fixed-step 4th-order Runge-Kutta (default)
            - 'euler': Fixed-step forward Euler
            - 'adaptive': Use same adaptive solver for backward integration,
              automatically matching forward/backward accuracy
            - 'binomial': Use binomial checkpointing (Revolve algorithm) for
              O(log n) memory with O(n log n) recomputation
            - 'backsolve': Recompute forward trajectory during backward pass
              for true O(1) memory. Only stable for non-chaotic systems.
            - 'implicit': Use implicit BDF-1 (Backward Euler) for stiff systems.
              Handles stiff adjoint ODEs that arise from stiff forward problems.
        - 'n_steps': int, number of integration steps for fixed-step methods
          (rk4, euler, implicit). Default: 100. Ignored for 'adaptive' method.
        - 'gradient_clip': float, optional. Threshold for clipping adjoint
          gradients to prevent explosion in unstable/chaotic systems.
          None (default) disables clipping.
        - 'gradient_clip_mode': str, clipping mode. Options:
            - 'norm': Rescale adjoint vector if norm exceeds threshold (default)
            - 'value': Element-wise clamping to [-gradient_clip, gradient_clip]

    Returns
    -------
    wrapped_solver : callable
        Solver with same signature but using adjoint gradients.

    Examples
    --------
    >>> from torchscience.ordinary_differential_equation.initial_value_problem import (
    ...     adjoint,
    ...     dormand_prince_5,
    ... )
    >>> adjoint_solver = adjoint(dormand_prince_5)
    >>> y_final, interp = adjoint_solver(f, y0, t_span=(0.0, 10.0))
    >>> loss = y_final.sum()
    >>> loss.backward()  # Uses O(1) memory for autograd graph

    Notes
    -----
    The adjoint method only affects gradient computation. It does NOT change:

    - The interpolant (still stores trajectory points for dense output)
    - Forward solve behavior (same numerical solution)
    - Return values (same (y_final, interp) tuple)

    Memory savings come from not storing the autograd computation graph,
    not from discarding the trajectory.

    When to use:

    - Long integrations with large state dimension
    - Memory-constrained environments
    - When exact discretization gradients are not required

    When NOT to use:

    - Short integrations (overhead not worth it)
    - When you need exact discretization gradients
    - When differentiating through the interpolant
    """

    def wrapped_solver(
        f: Callable,
        y0: Union[torch.Tensor, TensorDict],
        t_span: Tuple[float, float],
        **kwargs,
    ) -> Tuple[
        Union[torch.Tensor, TensorDict],
        Callable[
            [Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]
        ],
    ]:
        """Wrapped solver using adjoint method for gradients."""
        t0, t1 = t_span

        # Handle TensorDict
        is_tensordict = isinstance(y0, TensorDict)
        y0_flat, unflatten = flatten_state(y0)

        if is_tensordict:

            def f_flat(t, y):
                y_struct = unflatten(y)
                dy_struct = f(t, y_struct)
                dy_flat, _ = flatten_state(dy_struct)
                return dy_flat

        else:
            f_flat = f

        # Determine parameters for gradient computation
        # If explicit params provided, use them; otherwise extract from closure
        if params is not None:
            # Use explicit params (may be empty list for forward-only)
            effective_params = list(params)
        else:
            # Legacy behavior: extract parameters from dynamics function closure
            # This is tricky - we need to capture them from f's closure
            # For now, we rely on the dynamics function using module parameters
            effective_params = []

            def extract_params(obj, seen=None):
                """Recursively extract tensors requiring grad from closures."""
                if seen is None:
                    seen = set()
                if id(obj) in seen:
                    return
                seen.add(id(obj))

                if isinstance(obj, torch.Tensor) and obj.requires_grad:
                    effective_params.append(obj)
                elif hasattr(obj, "__closure__") and obj.__closure__:
                    for cell in obj.__closure__:
                        try:
                            extract_params(cell.cell_contents, seen)
                        except ValueError:
                            pass
                elif hasattr(obj, "__dict__"):
                    for v in obj.__dict__.values():
                        extract_params(v, seen)

            extract_params(f)

        # If no parameters found, just do regular forward (no gradients needed)
        if not effective_params:
            # If params=[] was explicitly provided, disable gradients entirely
            # (forward-only solve). Otherwise, allow regular autograd.
            use_no_grad = params is not None and len(params) == 0
            context = torch.no_grad() if use_no_grad else torch.enable_grad()

            with context:
                y_final, interp = solver(
                    f_flat, y0_flat, t_span=(t0, t1), **kwargs
                )
            if is_tensordict:
                y_final = unflatten(y_final)

                def interp_td(t_query):
                    y_flat_query = interp(t_query)
                    if isinstance(t_query, (int, float)) or (
                        isinstance(t_query, torch.Tensor)
                        and t_query.dim() == 0
                    ):
                        return unflatten(y_flat_query)
                    return torch.stack(
                        [
                            unflatten(y_flat_query[i])
                            for i in range(y_flat_query.shape[0])
                        ]
                    )

                interp_td.success = getattr(interp, "success", None)
                return y_final, interp_td
            return y_final, interp

        # Use custom autograd function for adjoint gradients
        # Pass params as *args so they're tracked by autograd
        # Use holder to retrieve interpolant without a second forward solve
        interp_holder = _InterpolantHolder()

        # Determine adjoint method
        adj_method = (adjoint_options or {}).get("method", "rk4")

        if adj_method == "adaptive":
            # Use adaptive adjoint (same solver for backward integration)
            y_final_flat = _AdaptiveAdjointFunction.apply(
                y0_flat,
                t0,
                t1,
                solver,
                f_flat,
                kwargs,
                adjoint_options,
                interp_holder,
                *effective_params,
            )
        elif adj_method == "backsolve":
            # Use backsolve adjoint (recompute forward during backward)
            y_final_flat = _BacksolveAdjointFunction.apply(
                y0_flat,
                t0,
                t1,
                solver,
                f_flat,
                kwargs,
                adjoint_options,
                interp_holder,
                *effective_params,
            )
        elif adj_method == "binomial":
            # Use binomial checkpointing (Revolve algorithm)
            y_final_flat = _BinomialAdjointFunction.apply(
                y0_flat,
                t0,
                t1,
                solver,
                f_flat,
                kwargs,
                adjoint_options,
                interp_holder,
                *effective_params,
            )
        elif adj_method == "implicit":
            # Use implicit BDF adjoint for stiff systems
            y_final_flat = _ImplicitAdjointFunction.apply(
                y0_flat,
                t0,
                t1,
                solver,
                f_flat,
                kwargs,
                adjoint_options,
                interp_holder,
                *effective_params,
            )
        elif checkpoints is not None and checkpoints > 0:
            # Use checkpointed adjoint for memory/compute tradeoff
            y_final_flat = _CheckpointedAdjointFunction.apply(
                y0_flat,
                t0,
                t1,
                solver,
                f_flat,
                kwargs,
                adjoint_options,
                checkpoints,
                interp_holder,
                *effective_params,
            )
        else:
            # Use standard adjoint (store full trajectory interpolant)
            y_final_flat = _AdjointODEFunction.apply(
                y0_flat,
                t0,
                t1,
                solver,
                f_flat,
                kwargs,
                adjoint_options,
                interp_holder,
                *effective_params,
            )

        # Retrieve interpolant from holder (populated during forward pass)
        interp = interp_holder.interp

        # Unflatten result
        if is_tensordict:
            y_final = unflatten(y_final_flat)

            def interp_tensordict(t_query):
                y_flat_query = interp(t_query)
                if isinstance(t_query, (int, float)) or (
                    isinstance(t_query, torch.Tensor) and t_query.dim() == 0
                ):
                    return unflatten(y_flat_query)
                return torch.stack(
                    [
                        unflatten(y_flat_query[i])
                        for i in range(y_flat_query.shape[0])
                    ]
                )

            interp_tensordict.success = getattr(interp, "success", None)
            return y_final, interp_tensordict

        return y_final_flat, interp

    # Preserve solver metadata
    wrapped_solver.__name__ = (
        f"adjoint({getattr(solver, '__name__', 'solver')})"
    )
    wrapped_solver.__doc__ = f"""
    {getattr(solver, "__name__", "Solver")} with adjoint method for memory-efficient gradients.

    This is a wrapped version of {getattr(solver, "__name__", "the solver")} that uses
    the continuous adjoint method to compute gradients with O(1) memory instead of
    O(n_steps).

    See `adjoint()` documentation for details.
    """

    return wrapped_solver

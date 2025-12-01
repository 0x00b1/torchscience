"""Continuous Normalizing Flow utilities.

Provides trace estimation and dynamics wrappers for CNFs.
"""

from typing import Callable, Literal

import torch


def hutchinson_trace(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    n_samples: int = 10,
    noise_type: Literal["rademacher", "gaussian"] = "rademacher",
) -> torch.Tensor:
    """
    Estimate trace of Jacobian using Hutchinson's estimator.

    trace(J) = E[v^T J v] where v is random noise

    Parameters
    ----------
    f : callable
        Function f: R^n -> R^n.
    x : Tensor
        Point at which to estimate trace.
    n_samples : int
        Number of random samples.
    noise_type : str
        Type of noise: 'rademacher' (+/-1) or 'gaussian'.

    Returns
    -------
    trace : Tensor
        Estimated trace of Jacobian.
    """
    n = x.numel()
    trace_sum = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    x_flat = x.flatten()

    for _ in range(n_samples):
        if noise_type == "rademacher":
            v = (
                torch.randint(0, 2, (n,), device=x.device, dtype=x.dtype) * 2
                - 1
            )
        else:  # gaussian
            v = torch.randn(n, device=x.device, dtype=x.dtype)

        x_input = x_flat.clone().requires_grad_(True)
        f_val = f(x_input.reshape(x.shape)).flatten()

        # v^T J v = v^T @ grad(v^T f)
        vjp = torch.autograd.grad(
            f_val,
            x_input,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
        )[0]

        trace_sum = trace_sum + torch.dot(v, vjp)

    return trace_sum / n_samples


def exact_trace(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute exact trace of Jacobian (expensive for large n).

    Parameters
    ----------
    f : callable
        Function f: R^n -> R^n.
    x : Tensor
        Point at which to compute trace.

    Returns
    -------
    trace : Tensor
        Exact trace of Jacobian.
    """
    n = x.numel()
    x_flat = x.flatten().requires_grad_(True)
    f_val = f(x_flat.reshape(x.shape)).flatten()

    trace = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    for i in range(n):
        grad_i = torch.autograd.grad(
            f_val[i],
            x_flat,
            retain_graph=(i < n - 1),
            create_graph=True,
        )[0]
        trace = trace + grad_i[i]

    return trace


def cnf_dynamics(
    velocity: Callable[[float, torch.Tensor], torch.Tensor],
    trace_method: Literal["exact", "hutchinson"] = "hutchinson",
    n_trace_samples: int = 10,
) -> Callable[[float, torch.Tensor], torch.Tensor]:
    """
    Wrap velocity field for CNF integration with log-probability tracking.

    The augmented state is [z, log_p] where:
    - dz/dt = velocity(t, z)
    - d(log_p)/dt = -div(velocity) = -trace(d velocity / dz)

    Parameters
    ----------
    velocity : callable
        Velocity field v(t, z) -> dz/dt.
    trace_method : str
        Method for trace estimation: 'exact' or 'hutchinson'.
    n_trace_samples : int
        Number of samples for Hutchinson estimator.

    Returns
    -------
    cnf_f : callable
        Augmented dynamics for [z, log_p] state.
    """

    def cnf_f(t: float, state: torch.Tensor) -> torch.Tensor:
        z = state[:-1]

        # Compute velocity
        v = velocity(t, z)

        # Compute divergence (negative trace)
        def v_func(z_input):
            return velocity(t, z_input)

        if trace_method == "exact":
            div = exact_trace(v_func, z)
        else:
            div = hutchinson_trace(v_func, z, n_samples=n_trace_samples)

        # d(log_p)/dt = -div(v)
        d_log_p = -div

        return torch.cat([v.flatten(), d_log_p.unsqueeze(0)])

    return cnf_f

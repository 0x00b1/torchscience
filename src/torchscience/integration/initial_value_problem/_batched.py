"""Native batched ODE solving for efficient parallel integration."""

from typing import Callable, Tuple

from torch import Tensor

from torchscience.integration.initial_value_problem._dormand_prince_5 import (
    dormand_prince_5,
)


def solve_ivp_batched(
    f: Callable[[float, Tensor], Tensor],
    y0: Tensor,
    t_span: Tuple[float, float],
    *,
    method: str = "dormand_prince_5",
    step_strategy: str = "synchronized",
    **kwargs,
) -> Tuple[Tensor, Callable]:
    """
    Batched ODE solving with explicit batch dimension.

    Parameters
    ----------
    f : callable
        Dynamics function f(t, y) where y has shape (batch, state_dim).
        Must be vectorized over batch dimension.
    y0 : Tensor
        Initial state, shape (batch, state_dim).
    t_span : tuple of float
        Integration interval (t0, t1).
    method : str
        Solver method (default: "dormand_prince_5").
    step_strategy : str
        - "synchronized": All batch elements use same step size.

    Returns
    -------
    y_final : Tensor
        Final state, shape (batch, state_dim).
    interp : callable
        Interpolant function.
    """
    if step_strategy != "synchronized":
        raise NotImplementedError(
            f"step_strategy='{step_strategy}' not yet implemented"
        )

    # For synchronized stepping, we can use the standard solver
    # by treating the batch as a single larger state
    batch_size, state_dim = y0.shape
    y0_flat = y0.reshape(-1)

    def f_flat(t, y_flat):
        y_batched = y_flat.reshape(batch_size, state_dim)
        dy_batched = f(t, y_batched)
        return dy_batched.reshape(-1)

    if method == "dormand_prince_5":
        y_final_flat, interp_flat = dormand_prince_5(
            f_flat, y0_flat, t_span, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    y_final = y_final_flat.reshape(batch_size, state_dim)

    def interp_batched(t):
        return interp_flat(t).reshape(batch_size, state_dim)

    return y_final, interp_batched

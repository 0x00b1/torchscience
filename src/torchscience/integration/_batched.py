"""Native batched ODE solving for efficient parallel integration."""

from typing import Callable, List, Optional, Tuple

from torch import Tensor

from torchscience.integration._dormand_prince_5 import (
    dormand_prince_5,
)
from torchscience.integration._ivp_adjoint import adjoint


def solve_ivp_batched(
    f: Callable[[float, Tensor], Tensor],
    y0: Tensor,
    t_span: Tuple[float, float],
    *,
    method: str = "dormand_prince_5",
    step_strategy: str = "synchronized",
    sensitivity: Optional[str] = None,
    params: Optional[List[Tensor]] = None,
    adjoint_options: Optional[dict] = None,
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
    sensitivity : str, optional
        Sensitivity method for gradient computation:
        - None: Use standard autograd (default)
        - "adjoint": Use adjoint method for memory-efficient gradients
    params : list of Tensor, optional
        Parameters to compute gradients for when using adjoint sensitivity.
    adjoint_options : dict, optional
        Options for adjoint integration (see adjoint() for details).

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

    # Select the base solver
    if method == "dormand_prince_5":
        base_solver = dormand_prince_5
    else:
        raise ValueError(f"Unknown method: {method}")

    # Wrap with adjoint if requested
    if sensitivity == "adjoint":
        solver = adjoint(
            base_solver, params=params, adjoint_options=adjoint_options
        )
        y_final_flat, interp_flat = solver(
            f_flat, y0_flat, t_span=t_span, **kwargs
        )
    elif sensitivity is None:
        y_final_flat, interp_flat = base_solver(
            f_flat, y0_flat, t_span, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown sensitivity method: {sensitivity}. Use None or 'adjoint'."
        )

    y_final = y_final_flat.reshape(batch_size, state_dim)

    def interp_batched(t):
        return interp_flat(t).reshape(batch_size, state_dim)

    return y_final, interp_batched

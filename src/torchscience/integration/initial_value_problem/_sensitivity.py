"""Sensitivity analysis utilities for ODE solutions."""

from typing import Callable, List, Optional, Tuple, Union

from torch import Tensor

from torchscience.integration.initial_value_problem._adjoint import adjoint
from torchscience.integration.initial_value_problem._dormand_prince_5 import (
    dormand_prince_5,
)


def solve_ivp_sensitivity(
    f: Callable[[float, Tensor], Tensor],
    y0: Tensor,
    t_span: Tuple[float, float],
    *,
    params: List[Tensor],
    mode: str = "gradient",
    loss_fn: Optional[Callable[[Tensor], Tensor]] = None,
    method: str = "dormand_prince_5",
    **kwargs,
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """
    Compute sensitivity measures for ODE solutions.

    Parameters
    ----------
    f : callable
        Dynamics function f(t, y).
    y0 : Tensor
        Initial state.
    t_span : tuple of float
        Integration interval (t0, t1).
    params : list of Tensor
        Parameters to compute sensitivities for.
    mode : str
        Sensitivity mode:
        - "gradient": dL/dtheta (requires loss_fn)
        - "jacobian": dy_final/dtheta (full Jacobian)
        - "fisher": Fisher information matrix
    loss_fn : callable, optional
        Loss function L(y_final) -> scalar. Required for mode="gradient".
    method : str
        ODE solver method.
    **kwargs
        Additional arguments for solver.

    Returns
    -------
    sensitivity : Tensor
        - mode="gradient": gradient tensor of shape (param_dim,)
        - mode="jacobian": Jacobian tensor of shape (state_dim, param_dim)
        - mode="fisher": Fisher matrix of shape (param_dim, param_dim)
    """
    if mode == "gradient":
        if loss_fn is None:
            raise ValueError("loss_fn required for mode='gradient'")
        return _compute_gradient(
            f, y0, t_span, params, loss_fn, method, **kwargs
        )
    elif mode == "jacobian":
        return _compute_jacobian(f, y0, t_span, params, method, **kwargs)
    elif mode == "fisher":
        return _compute_fisher(f, y0, t_span, params, method, **kwargs)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use 'gradient', 'jacobian', or 'fisher'."
        )


def _compute_gradient(f, y0, t_span, params, loss_fn, method, **kwargs):
    """Compute gradient dL/dtheta using adjoint method."""
    # Clone params and enable gradients
    cloned_params = []
    for p in params:
        p_clone = p.clone().requires_grad_(True)
        cloned_params.append(p_clone)

    # Get solver
    if method == "dormand_prince_5":
        solver = dormand_prince_5
    else:
        raise ValueError(f"Unknown method: {method}")

    # Wrap with adjoint for memory-efficient gradients
    adjoint_solver = adjoint(solver, params=cloned_params)

    # Solve ODE
    y_final, _ = adjoint_solver(f, y0, t_span=t_span, **kwargs)

    # Compute loss and backpropagate
    loss = loss_fn(y_final)
    loss.backward()

    # Collect gradients
    if len(cloned_params) == 1:
        return cloned_params[0].grad
    else:
        return tuple(p.grad for p in cloned_params)


def _compute_jacobian(f, y0, t_span, params, method, **kwargs):
    """Compute Jacobian dy_final/dtheta."""
    raise NotImplementedError("Jacobian mode not yet implemented")


def _compute_fisher(f, y0, t_span, params, method, **kwargs):
    """Compute Fisher information matrix."""
    raise NotImplementedError("Fisher mode not yet implemented")

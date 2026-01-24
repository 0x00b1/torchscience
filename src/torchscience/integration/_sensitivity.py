"""Sensitivity analysis utilities for ODE solutions."""

from typing import Callable, List, Optional, Tuple, Union

from torch import Tensor

from torchscience.integration._dormand_prince_5 import (
    dormand_prince_5,
)
from torchscience.integration._ivp_adjoint import adjoint


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
    """Compute Jacobian dy_final/dtheta.

    The Jacobian is computed by solving the ODE with gradient tracking,
    then computing the gradient of each output component with respect to
    all parameters.

    For dy/dt = f(t, y; theta), the Jacobian J = dy(T)/dtheta has shape
    (state_dim, param_dim) where each row i contains the gradient of
    y_final[i] with respect to all parameters.

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
    method : str
        ODE solver method.
    **kwargs
        Additional arguments for solver.

    Returns
    -------
    J : Tensor
        Jacobian tensor of shape (state_dim, total_param_dim).
    """
    import torch

    # Get solver
    if method == "dormand_prince_5":
        solver = dormand_prince_5
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute total parameter dimension
    param_dims = [p.numel() for p in params]
    total_param_dim = sum(param_dims)
    state_dim = y0.numel()

    # Initialize Jacobian matrix
    J = torch.zeros(
        state_dim, total_param_dim, dtype=y0.dtype, device=y0.device
    )

    # We need to compute gradients through the adjoint solver.
    # The key issue is that the dynamics function `f` captures the original params
    # from its closure. The adjoint solver will use these original params.
    #
    # To compute the full Jacobian, we use repeated backward passes, once for
    # each output dimension, using the loss function approach from gradient mode.

    # Ensure params require grad
    for p in params:
        p.requires_grad_(True)

    # Wrap with adjoint for gradient computation
    adjoint_solver = adjoint(solver, params=params)

    # Solve ODE once
    y_final, _ = adjoint_solver(f, y0, t_span=t_span, **kwargs)

    # For each output dimension, compute gradient w.r.t. all params
    y_final_flat = y_final.flatten()

    for i in range(state_dim):
        # Zero out any existing gradients
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        # Backward pass for the i-th output
        # Create a one-hot selector for output i
        retain = i < state_dim - 1
        y_final_flat[i].backward(retain_graph=retain)

        # Collect gradients into Jacobian row
        offset = 0
        for p in params:
            if p.grad is not None:
                J[i, offset : offset + p.numel()] = p.grad.flatten().clone()
            offset += p.numel()

    return J


def _compute_fisher(f, y0, t_span, params, method, **kwargs):
    """Compute Fisher information matrix F = J^T @ J.

    The Fisher information matrix provides a measure of how much information
    the ODE solution carries about the parameters. It is useful for uncertainty
    quantification via the Cramér-Rao bound.

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
    method : str
        ODE solver method.
    **kwargs
        Additional arguments for solver.

    Returns
    -------
    fisher : Tensor
        Fisher information matrix of shape (param_dim, param_dim).
    """
    # First compute Jacobian
    J = _compute_jacobian(f, y0, t_span, params, method, **kwargs)

    # Fisher = J^T @ J
    fisher = J.T @ J

    return fisher

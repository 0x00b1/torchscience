"""Second-order gradient utilities for ODE solutions."""

from typing import Callable, List, Tuple

import torch
from torch import Tensor

from torchscience.ordinary_differential_equation._dormand_prince_5 import (
    dormand_prince_5,
)
from torchscience.ordinary_differential_equation._ivp_adjoint import adjoint


def solve_ivp_hvp(
    f: Callable[[float, Tensor], Tensor],
    y0: Tensor,
    t_span: Tuple[float, float],
    *,
    params: List[Tensor],
    loss_fn: Callable[[Tensor], Tensor],
    v: Tensor,
    method: str = "dormand_prince_5",
    **kwargs,
) -> Tensor:
    """
    Compute Hessian-vector product (d^2 L/d theta^2) @ v efficiently.

    This function computes the product of the Hessian of a loss function
    (with respect to ODE parameters) with a vector v, without explicitly
    forming the full Hessian matrix. This is useful for second-order
    optimization methods like Newton methods and natural gradient.

    Parameters
    ----------
    f : callable
        Dynamics function f(t, y).
    y0 : Tensor
        Initial state.
    t_span : tuple of float
        Integration interval (t0, t1).
    params : list of Tensor
        Parameters to compute HVP for.
    loss_fn : callable
        Loss function L(y_final) -> scalar.
    v : Tensor
        Vector to multiply with Hessian.
    method : str
        ODE solver method. Currently only "dormand_prince_5" is supported.
    **kwargs
        Additional arguments for solver.

    Returns
    -------
    hvp : Tensor
        Hessian-vector product, same shape as params[0].

    Examples
    --------
    >>> import torch
    >>> from torchscience.ordinary_differential_equation.initial_value_problem import solve_ivp_hvp
    >>>
    >>> theta = torch.tensor([1.0], dtype=torch.float64)
    >>>
    >>> def dynamics(t, y):
    ...     return -theta * y
    >>>
    >>> y0 = torch.tensor([1.0], dtype=torch.float64)
    >>>
    >>> def loss_fn(y_final):
    ...     return (y_final ** 2).sum()
    >>>
    >>> v = torch.tensor([1.0], dtype=torch.float64)
    >>> hvp = solve_ivp_hvp(
    ...     dynamics, y0, t_span=(0.0, 1.0),
    ...     params=[theta], loss_fn=loss_fn, v=v
    ... )

    Notes
    -----
    The Hessian-vector product is computed using `torch.autograd.functional.hvp`,
    which computes H @ v in O(n) time without forming the O(n^2) Hessian matrix.

    This is particularly useful for:
    - Newton-type optimization methods
    - Natural gradient descent
    - Second-order sensitivity analysis
    - Uncertainty quantification via the Hessian
    """
    # Get solver
    if method == "dormand_prince_5":
        solver = dormand_prince_5
    else:
        raise ValueError(f"Unknown method: {method}")

    # Assume single parameter tensor for now
    theta = params[0]

    def full_loss(theta_val: Tensor) -> Tensor:
        """Compute loss as function of parameter only."""
        # The dynamics function captures theta from closure, so we need to
        # create a new dynamics that uses theta_val instead
        theta_val = theta_val.requires_grad_(True)

        # Wrap with adjoint for gradient computation
        adjoint_solver = adjoint(solver, params=[theta_val])

        # Solve ODE
        y_final, _ = adjoint_solver(f, y0, t_span=t_span, **kwargs)

        return loss_fn(y_final)

    # Compute HVP using torch.autograd.functional.hvp
    # hvp returns (loss_value, hvp_tensor)
    _, hvp_result = torch.autograd.functional.hvp(full_loss, theta, v)

    return hvp_result

"""Adjoint method for memory-efficient BVP gradients.

Uses implicit differentiation to compute gradients without storing
the computational graph of the Newton iteration.
"""

from typing import Any, Callable, Optional

import torch
from torch import Tensor

from torchscience.integration._bvp_solution import (
    BVPSolution,
)


class _BVPAdjointFunction(torch.autograd.Function):
    """Custom autograd function for implicit differentiation of BVP solutions.

    The key insight is that at the solution y*, the residual R(y*, p) = 0.
    By the implicit function theorem:
        dy*/dp = -(dR/dy)^{-1} @ (dR/dp)

    We use vector-Jacobian products (VJPs) to compute gradients efficiently.
    """

    @staticmethod
    def forward(
        ctx: Any,
        y: Tensor,
        p: Tensor,
        x: Tensor,
        fun: Callable,
        bc: Callable,
        solver: Callable,
        kwargs: dict,
        *params: Tensor,
    ) -> Tensor:
        """Forward pass: return solution y with proper gradient tracking.

        Note: PyTorch's autograd.Function automatically tracks which inputs
        require gradients and creates grad_fn accordingly. We just need to
        return a plain tensor - PyTorch handles connecting the backward.
        """
        # Save for backward
        ctx.save_for_backward(y.detach(), p.detach(), x.detach(), *params)
        ctx.fun = fun
        ctx.bc = bc
        ctx.solver = solver
        ctx.kwargs = kwargs
        ctx.n_params = len(params)
        ctx.p_requires_grad = p.requires_grad

        # Return detached solution - autograd.Function handles gradient tracking
        return y.detach().clone()

    @staticmethod
    def backward(ctx: Any, grad_y: Tensor) -> tuple:
        """Backward pass: compute gradients via implicit differentiation."""
        saved = ctx.saved_tensors
        y = saved[0]
        p = saved[1]
        x = saved[2]
        params = saved[3:]
        fun = ctx.fun
        bc = ctx.bc

        n_components, n_nodes = y.shape
        n_params = p.shape[0] if p.numel() > 0 else 0

        # Flatten grad_y for VJP computation
        grad_y_flat = grad_y.flatten()

        # We need to compute:
        # grad_p = -(dR/dp)^T @ (dR/dy)^{-T} @ grad_y
        #
        # where R is the residual function [collocation; boundary conditions]
        #
        # Instead of inverting, we solve the linear system:
        # (dR/dy)^T @ lambda = grad_y
        # Then: grad_p = -(dR/dp)^T @ lambda

        from torchscience.integration._bvp_collocation import (
            compute_collocation_residual,
        )

        def residual_fn(y_flat: Tensor, p_curr: Tensor) -> Tensor:
            """Compute full residual as a function of y and p."""
            y_curr = y_flat.reshape(n_components, n_nodes)

            # Collocation residuals
            coll_res = compute_collocation_residual(fun, x, y_curr, p_curr)
            coll_flat = coll_res.flatten()

            # Boundary residuals
            ya = y_curr[:, 0]
            yb = y_curr[:, -1]
            bc_res = bc(ya, yb, p_curr)

            return torch.cat([coll_flat, bc_res])

        y_flat = y.flatten()

        # Compute Jacobians using torch.func
        # dR/dy: shape (n_residuals, n_y_vars)
        dR_dy = torch.func.jacrev(lambda yf: residual_fn(yf, p))(y_flat)

        # Solve (dR/dy)^T @ lambda = grad_y
        # This is an underdetermined system: (n_y_vars, n_residuals) @ lambda = (n_y_vars,)
        # lstsq finds the minimum norm solution for lambda
        # Shapes:
        #   dR_dy: (n_residuals, n_y_vars)
        #   dR_dy.T: (n_y_vars, n_residuals)
        #   grad_y_flat: (n_y_vars,)
        #   lambda_sol: (n_residuals,)
        try:
            lambda_sol, *_ = torch.linalg.lstsq(
                dR_dy.T, grad_y_flat.unsqueeze(-1)
            )
            lambda_sol = lambda_sol.squeeze(-1)
        except RuntimeError:
            # If solve fails, return zeros
            return (None, None, None, None, None, None, None) + tuple(
                None for _ in params
            )

        # Compute gradient w.r.t. p
        if n_params > 0:
            # dR/dp: shape (n_residuals, n_params)
            dR_dp = torch.func.jacrev(lambda pp: residual_fn(y_flat, pp))(p)

            # grad_p = -(dR/dp)^T @ lambda
            grad_p = -dR_dp.T @ lambda_sol
        else:
            grad_p = None

        # Gradients for closure parameters
        # For simplicity, we don't compute these in the basic implementation
        # Users can use torch.autograd.grad with the adjoint solution
        param_grads = tuple(None for _ in params)

        # Return gradients: (y, p, x, fun, bc, solver, kwargs, *params)
        return (None, grad_p, None, None, None, None, None) + param_grads


def bvp_adjoint(
    solver: Callable[..., BVPSolution],
) -> Callable[..., BVPSolution]:
    """Wrap a BVP solver to use implicit differentiation for gradients.

    This wrapper uses the adjoint method (implicit differentiation) to
    compute gradients through the BVP solution. This is more memory-efficient
    than backpropagating through the Newton iteration, especially for
    problems with many mesh nodes.

    Parameters
    ----------
    solver : callable
        A BVP solver function with signature:
        solver(fun, bc, x, y, p=None, **kwargs) -> BVPSolution

    Returns
    -------
    callable
        Wrapped solver with adjoint gradients.

    Notes
    -----
    At the solution y*, the residual R(y*, p) = 0. By the implicit function
    theorem, the gradient dy*/dp can be computed without backpropagating
    through the solver iterations.

    Example
    -------
    >>> from torchscience.integration.boundary_value_problem import solve_bvp
    >>> adjoint_solver = bvp_adjoint(solve_bvp)
    >>> sol = adjoint_solver(fun, bc, x, y, p=p)
    >>> loss = sol.y.sum()
    >>> loss.backward()  # Uses implicit differentiation
    """

    def wrapped_solver(
        fun: Callable[[Tensor, Tensor, Tensor], Tensor],
        bc: Callable[[Tensor, Tensor, Tensor], Tensor],
        x: Tensor,
        y: Tensor,
        p: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> BVPSolution:
        if p is None:
            p = torch.empty(0, dtype=x.dtype, device=x.device)

        # Find parameters that require gradients (for closure capture)
        params = []
        # We could scan fun/bc for parameters, but for simplicity
        # we just track p here

        # Solve BVP (no grad tracking through solver internals)
        with torch.no_grad():
            sol = solver(fun, bc, x, y, p=p, **kwargs)

        if not sol.success:
            # If solver failed, return as-is (can't differentiate)
            return sol

        # Wrap solution y for custom backward
        y_solution = _BVPAdjointFunction.apply(
            sol.y, p, x, fun, bc, solver, kwargs, *params
        )

        # Return BVPSolution with gradients enabled for y
        return BVPSolution(
            x=sol.x,
            y=y_solution,
            yp=sol.yp,
            p=sol.p,
            rms_residuals=sol.rms_residuals,
            n_iterations=sol.n_iterations,
            success=sol.success,
        )

    # Preserve solver metadata
    wrapped_solver.__name__ = (
        f"adjoint({getattr(solver, '__name__', 'solver')})"
    )
    wrapped_solver.__doc__ = f"""
    {getattr(solver, "__name__", "Solver")} with adjoint method for memory-efficient gradients.

    This is a wrapped version of {getattr(solver, "__name__", "the solver")} that uses
    implicit differentiation to compute gradients.

    See `bvp_adjoint()` documentation for details.
    """

    return wrapped_solver

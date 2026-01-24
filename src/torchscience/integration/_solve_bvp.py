"""Main BVP solver API."""

import warnings
from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.integration._bvp_collocation import (
    compute_collocation_residual,
    compute_collocation_residual_and_f,
)
from torchscience.integration._bvp_exceptions import (
    BVPConvergenceError,
    BVPMeshError,
)
from torchscience.integration._bvp_mesh import (
    compute_rms_residuals,
    refine_mesh,
)
from torchscience.integration._bvp_newton import newton_bvp
from torchscience.integration._bvp_solution import (
    BVPSolution,
)


def solve_bvp(
    fun: Callable[[Tensor, Tensor, Tensor], Tensor],
    bc: Callable[[Tensor, Tensor, Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    p: Optional[Tensor] = None,
    *,
    tol: float = 1e-3,
    max_nodes: int = 1000,
    max_outer_iterations: int = 10,
    throw: bool = True,
    verbose: int = 0,
) -> BVPSolution:
    """Solve a boundary value problem for a system of ODEs.

    This function solves a first-order system of ODEs:

        dy/dx = fun(x, y, p)

    subject to two-point boundary conditions:

        bc(y(a), y(b), p) = 0

    using a 4th-order Lobatto collocation method with adaptive mesh
    refinement.

    Parameters
    ----------
    fun : callable
        Right-hand side of the ODE system. Signature: fun(x, y, p) -> dy/dx
        where x has shape (n_points,), y has shape (n_components, n_points),
        and p has shape (k,). Must support vectorized evaluation.
    bc : callable
        Boundary condition residuals. Signature: bc(ya, yb, p) -> residual
        where ya, yb have shape (n_components,) and result has shape (n_bc,).
        For n_components equations with k unknown parameters, n_bc = n_components + k.
    x : Tensor
        Initial mesh, shape (m,). Must be strictly increasing.
        Defines the interval [x[0], x[-1]].
    y : Tensor
        Initial guess for solution, shape (n_components, m).
    p : Tensor, optional
        Initial guess for unknown parameters, shape (k,).
        Default is empty (no unknown parameters).
    tol : float
        Tolerance for residual. Default is 1e-3.
    max_nodes : int
        Maximum number of mesh nodes. Default is 1000.
    max_outer_iterations : int
        Maximum number of mesh adaptation iterations. Default is 10.
    throw : bool
        If True (default), raise exceptions on solver failures. If False, return
        BVPSolution with success=False instead of raising.
    verbose : int
        Verbosity level. 0 = silent, 1 = summary, 2 = per-iteration.
        Uses warnings.warn() for messages (not print).

    Returns
    -------
    BVPSolution
        Solution object with attributes:
        - x: final mesh
        - y: solution values
        - p: solved parameters
        - rms_residuals: final residual norm
        - n_iterations: total Newton iterations
        - success: whether solver converged (always check when throw=False)

    Raises
    ------
    BVPMeshError
        If mesh refinement exceeds max_nodes (only when throw=True).
    BVPConvergenceError
        If Newton iteration fails to converge (only when throw=True).

    Examples
    --------
    Solve y'' = -y with y(0) = 0, y(pi) = 0:

    >>> import torch
    >>> def fun(x, y, p):
    ...     return torch.stack([y[1], -y[0]])
    >>> def bc(ya, yb, p):
    ...     return torch.stack([ya[0], yb[0]])
    >>> x = torch.linspace(0, torch.pi, 10, dtype=torch.float64)
    >>> y = torch.stack([torch.sin(x), torch.cos(x)])
    >>> sol = solve_bvp(fun, bc, x, y)
    >>> sol.success
    True
    """
    if p is None:
        p = torch.empty(0, dtype=x.dtype, device=x.device)

    # Handle meta tensors (shape inference only)
    if x.device.type == "meta":
        from torchscience.integration._bvp_meta import (
            solve_bvp_meta,
        )

        return solve_bvp_meta(
            fun, bc, x, y, p, tol, max_nodes, max_outer_iterations
        )

    y_current = y.clone()
    x_current = x.clone()
    p_current = p.clone()

    total_iterations = 0

    for outer_iter in range(max_outer_iterations):
        # Solve on current mesh
        y_new, p_new, converged, n_iter = newton_bvp(
            fun, bc, x_current, y_current, p_current, tol=tol * 0.1
        )
        total_iterations += n_iter

        if not converged:
            if throw:
                raise BVPConvergenceError(
                    f"Newton iteration failed to converge on iteration {outer_iter}"
                )
            if verbose > 0:
                warnings.warn(
                    f"Newton did not converge on iteration {outer_iter}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            # Return failed solution
            coll_res = compute_collocation_residual(
                fun, x_current, y_current, p_current
            )
            rms_res = compute_rms_residuals(coll_res)
            return BVPSolution(
                x=x_current,
                y=y_current,
                yp=fun(x_current, y_current, p_current),
                p=p_current,
                rms_residuals=rms_res.max(),
                n_iterations=total_iterations,
                success=False,
            )

        y_current = y_new
        p_current = p_new

        # Compute residuals and derivatives for mesh adaptation
        coll_res, f_left, f_mid, f_right = compute_collocation_residual_and_f(
            fun, x_current, y_current, p_current
        )
        rms_res = compute_rms_residuals(coll_res)

        if verbose > 1:
            warnings.warn(
                f"Iteration {outer_iter}: max residual = {rms_res.max():.2e}",
                RuntimeWarning,
                stacklevel=2,
            )

        # Check if tolerance is met
        if rms_res.max() < tol:
            if verbose > 0:
                warnings.warn(
                    f"Converged in {total_iterations} iterations",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return BVPSolution(
                x=x_current,
                y=y_current,
                yp=fun(x_current, y_current, p_current),
                p=p_current,
                rms_residuals=rms_res.max(),
                n_iterations=total_iterations,
                success=True,
            )

        # Compute f at all nodes for interpolation
        f_nodes = fun(x_current, y_current, p_current)

        # Refine mesh using cubic Hermite interpolation
        x_new, y_new = refine_mesh(x_current, y_current, f_nodes, rms_res, tol)

        if x_new.shape[0] > max_nodes:
            if throw:
                raise BVPMeshError(x_new.shape[0], max_nodes)
            if verbose > 0:
                warnings.warn(
                    f"Mesh refinement would exceed max_nodes ({x_new.shape[0]} > {max_nodes})",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return BVPSolution(
                x=x_current,
                y=y_current,
                yp=fun(x_current, y_current, p_current),
                p=p_current,
                rms_residuals=rms_res.max(),
                n_iterations=total_iterations,
                success=False,
            )

        if x_new.shape[0] == x_current.shape[0]:
            # No refinement happened but tolerance not met
            if verbose > 0:
                warnings.warn(
                    "No refinement possible, returning current solution",
                    RuntimeWarning,
                    stacklevel=2,
                )
            break

        x_current = x_new
        y_current = y_new

    # Return best solution even if not fully converged
    coll_res = compute_collocation_residual(
        fun, x_current, y_current, p_current
    )
    rms_res = compute_rms_residuals(coll_res)

    return BVPSolution(
        x=x_current,
        y=y_current,
        yp=fun(x_current, y_current, p_current),
        p=p_current,
        rms_residuals=rms_res.max(),
        n_iterations=total_iterations,
        success=rms_res.max() < tol,
    )

from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult


def nelder_mead(
    fun: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    maxiter: Optional[int] = None,
    tol: Optional[float] = None,
    initial_simplex: Optional[Tensor] = None,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
) -> OptimizeResult:
    r"""
    Nelder-Mead simplex method for derivative-free optimization.

    Minimizes a scalar-valued function without using gradient information.
    The method maintains a simplex of ``n+1`` vertices in ``n``-dimensional
    space and iteratively replaces the worst vertex using reflection,
    expansion, contraction, and shrink operations.

    Parameters
    ----------
    fun : Callable[[Tensor], Tensor]
        Scalar-valued objective function to minimize.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    maxiter : int, optional
        Maximum number of iterations. Default: ``200 * n``.
    tol : float, optional
        Convergence tolerance on simplex diameter. Default:
        ``torch.finfo(x0.dtype).eps ** 0.5``.
    initial_simplex : Tensor, optional
        Initial simplex of shape ``(n+1, n)``. If ``None``, constructed
        from ``x0`` using adaptive perturbations.
    alpha : float, optional
        Reflection coefficient. Default: 1.0.
    gamma : float, optional
        Expansion coefficient. Default: 2.0.
    rho : float, optional
        Contraction coefficient. Default: 0.5.
    sigma : float, optional
        Shrink coefficient. Default: 0.5.

    Returns
    -------
    OptimizeResult
        Result with ``x`` (best vertex), ``converged``, ``num_iterations``,
        ``fun``. Note: ``x`` is detached — no implicit differentiation
        (derivative-free method).
    """
    n = x0.numel()

    if maxiter is None:
        maxiter = 200 * n

    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    # Build initial simplex
    if initial_simplex is not None:
        simplex = initial_simplex.clone().detach()
    else:
        simplex = torch.zeros(n + 1, n, dtype=x0.dtype, device=x0.device)
        simplex[0] = x0.detach().clone()
        for i in range(n):
            vertex = x0.detach().clone()
            h = 0.05 if vertex[i] != 0 else 0.00025
            vertex[i] = vertex[i] + h
            simplex[i + 1] = vertex

    # Evaluate function at all vertices (ensure scalar values)
    def _eval(x: Tensor) -> Tensor:
        return fun(x).reshape(())

    with torch.no_grad():
        f_values = torch.stack([_eval(simplex[i]) for i in range(n + 1)])

    num_iter = 0

    for k in range(maxiter):
        num_iter = k + 1

        # Sort by function value
        order = torch.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]

        # Check convergence: simplex diameter
        diameter = (simplex[1:] - simplex[0]).abs().max()
        f_range = (f_values[-1] - f_values[0]).abs()
        if diameter < tol and f_range < tol:
            converged = torch.tensor(True, device=x0.device)
            break

        # Centroid of all vertices except worst
        centroid = simplex[:-1].mean(dim=0)

        # Reflection
        x_worst = simplex[-1]
        x_r = centroid + alpha * (centroid - x_worst)
        with torch.no_grad():
            f_r = _eval(x_r)

        if f_values[0] <= f_r < f_values[-2]:
            # Accept reflection
            simplex[-1] = x_r
            f_values[-1] = f_r
            continue

        if f_r < f_values[0]:
            # Try expansion
            x_e = centroid + gamma * (x_r - centroid)
            with torch.no_grad():
                f_e = _eval(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
            continue

        # Contraction
        if f_r < f_values[-1]:
            # Outside contraction
            x_c = centroid + rho * (x_r - centroid)
            with torch.no_grad():
                f_c = _eval(x_c)
            if f_c <= f_r:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue
        else:
            # Inside contraction
            x_c = centroid + rho * (x_worst - centroid)
            with torch.no_grad():
                f_c = _eval(x_c)
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue

        # Shrink
        best = simplex[0].clone()
        for i in range(1, n + 1):
            simplex[i] = best + sigma * (simplex[i] - best)
            with torch.no_grad():
                f_values[i] = _eval(simplex[i])
    else:
        # Final sort
        order = torch.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]
        diameter = (simplex[1:] - simplex[0]).abs().max()
        f_range = (f_values[-1] - f_values[0]).abs()
        converged = torch.tensor(
            diameter < tol and f_range < tol, device=x0.device
        )

    # Best vertex (detached — no implicit diff for derivative-free method)
    x_best = simplex[0].detach()
    f_best = f_values[0].detach()

    return OptimizeResult(
        x=x_best,
        converged=converged,
        num_iterations=torch.tensor(
            num_iter, dtype=torch.int64, device=x0.device
        ),
        fun=f_best,
    )

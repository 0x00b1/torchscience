from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._levenberg_marquardt import (
    levenberg_marquardt,
)


def least_squares(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    bounds: Optional[Tuple[Tensor, Tensor]] = None,
    tol: Optional[float] = None,
    maxiter: int = 100,
) -> OptimizeResult:
    r"""Solve a nonlinear least squares problem.

    Finds parameters :math:`x` that minimize the sum of squared residuals:

    .. math::

        \min_x \frac{1}{2} \|f(x)\|^2

    with optional box bounds :math:`\text{lb} \le x \le \text{ub}`.

    For the unbounded case, the Levenberg-Marquardt algorithm is applied
    directly. For the bounded case, a sigmoid variable transformation maps
    an unconstrained variable :math:`z` to the bounded domain
    :math:`x = \text{lb} + (\text{ub} - \text{lb}) \sigma(z)`, and the
    Levenberg-Marquardt algorithm is applied in :math:`z`-space.

    Parameters
    ----------
    fn : Callable[[Tensor], Tensor]
        Residual function. Takes parameters of shape ``(n,)`` and returns
        residuals of shape ``(m,)`` where ``m >= n``.
    x0 : Tensor
        Initial parameter guess of shape ``(n,)``.
    bounds : tuple of (Tensor, Tensor), optional
        Lower and upper bounds on the parameters. Each tensor has shape
        ``(n,)``. If None, the problem is unbounded.
    tol : float, optional
        Convergence tolerance on gradient norm. Default: ``sqrt(eps)`` for
        the dtype of ``x0``.
    maxiter : int
        Maximum number of iterations. Default: 100.

    Returns
    -------
    OptimizeResult
        Named tuple with fields:

        - **x** -- Solution tensor of shape ``(n,)``.
        - **converged** -- Boolean tensor indicating convergence.
        - **num_iterations** -- Number of iterations (set to ``maxiter``).
        - **fun** -- Objective value :math:`\frac{1}{2}\|f(x)\|^2` at the
          solution.

    Examples
    --------
    Solve a linear least squares problem:

    >>> def residuals(x):
    ...     return x - torch.tensor([1.0, 2.0])
    >>> result = least_squares(residuals, torch.zeros(2))
    >>> result.x
    tensor([1., 2.])

    Solve with box bounds:

    >>> def residuals(x):
    ...     return x - torch.tensor([5.0])
    >>> result = least_squares(
    ...     residuals,
    ...     torch.tensor([0.0]),
    ...     bounds=(torch.tensor([0.0]), torch.tensor([3.0])),
    ... )
    >>> result.x
    tensor([3.])
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    if bounds is not None:
        lb, ub = bounds
        lb = lb.to(dtype=x0.dtype, device=x0.device)
        ub = ub.to(dtype=x0.dtype, device=x0.device)

        # Clamp x0 into the interior of the bounds for computing initial z.
        # Use a margin of 1% of the bound width to avoid extreme z values
        # that would prevent the optimizer from moving.
        margin = 0.01 * (ub - lb)
        x0_clamped = torch.clamp(x0, lb + margin, ub - margin)

        # Inverse sigmoid to get initial z from x0
        t = (x0_clamped - lb) / (ub - lb)
        z0 = torch.log(t / (1.0 - t))

        def _transform(z: Tensor) -> Tensor:
            return lb + (ub - lb) * torch.sigmoid(z)

        def _residuals_z(z: Tensor) -> Tensor:
            return fn(_transform(z))

        z_opt = levenberg_marquardt(
            _residuals_z,
            z0,
            tol=tol,
            maxiter=maxiter,
        )

        x_opt = _transform(z_opt)
    else:
        x_opt = levenberg_marquardt(
            fn,
            x0,
            tol=tol,
            maxiter=maxiter,
        )

    # Compute objective value
    with torch.no_grad():
        r = fn(x_opt.detach())
        objective = 0.5 * torch.sum(r**2)

    # Check convergence by gradient norm
    with torch.no_grad():
        J = torch.func.jacrev(fn)(x_opt.detach())
        if J.dim() == 1:
            J = J.unsqueeze(0)
        g = J.T @ r
        converged = torch.norm(g) < tol

    return OptimizeResult(
        x=x_opt,
        converged=converged,
        num_iterations=torch.tensor(maxiter, dtype=torch.int64),
        fun=objective,
    )

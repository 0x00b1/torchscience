from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._l_bfgs import (
    _compute_grad,
    _implicit_diff_step,
)
from torchscience.optimization.minimization._newton_cg import (
    _hessian_vector_product,
)


def _steihaug_cg(hvp_fn, grad, trust_radius, max_cg_iter):
    """Steihaug-CG: solve the trust-region subproblem approximately.

    Minimizes m(d) = g^T d + 0.5 d^T H d  subject to ||d|| <= trust_radius.
    Uses CG and truncates at the trust-region boundary or negative curvature.
    """
    n = grad.numel()
    d = torch.zeros_like(grad)
    r = grad.clone()
    p = -r.clone()

    r_dot_r = (r * r).sum()
    tol_sq = (1e-2 * grad.norm()) ** 2

    for _ in range(min(max_cg_iter, n)):
        if r_dot_r < tol_sq:
            break

        Hp = hvp_fn(p)
        pHp = (p * Hp).sum()

        # Negative curvature: step to trust-region boundary along p
        if pHp <= 0:
            # Find tau such that ||d + tau * p|| = trust_radius
            dd = (d * d).sum()
            dp = (d * p).sum()
            pp = (p * p).sum()
            discriminant = dp**2 - pp * (dd - trust_radius**2)
            tau = (-dp + torch.sqrt(discriminant.clamp(min=0))) / pp.clamp(
                min=1e-30
            )
            return d + tau * p

        alpha = r_dot_r / pHp
        d_new = d + alpha * p

        # Check trust-region boundary
        if d_new.norm() >= trust_radius:
            # Step to boundary
            dd = (d * d).sum()
            dp = (d * p).sum()
            pp = (p * p).sum()
            discriminant = dp**2 - pp * (dd - trust_radius**2)
            tau = (-dp + torch.sqrt(discriminant.clamp(min=0))) / pp.clamp(
                min=1e-30
            )
            return d + tau * p

        d = d_new
        r = r + alpha * Hp
        r_dot_r_new = (r * r).sum()
        beta = r_dot_r_new / r_dot_r.clamp(min=1e-30)
        p = -r + beta * p
        r_dot_r = r_dot_r_new

    return d


def trust_region(
    fun: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    grad: Optional[Callable[[Tensor], Tensor]] = None,
    maxiter: int = 100,
    tol: Optional[float] = None,
    max_cg_iter: int = 50,
    initial_trust_radius: float = 1.0,
    max_trust_radius: float = 100.0,
    eta: float = 0.15,
) -> OptimizeResult:
    r"""
    Trust-region Newton method for unconstrained optimization.

    Uses the Steihaug-CG algorithm to solve the trust-region subproblem
    at each iteration. The trust-region radius is adjusted based on the
    ratio of actual to predicted reduction.

    Parameters
    ----------
    fun : Callable[[Tensor], Tensor]
        Scalar-valued objective function to minimize.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    grad : Callable[[Tensor], Tensor], optional
        Gradient function. If ``None``, uses ``torch.autograd``.
    maxiter : int, optional
        Maximum number of iterations. Default: 100.
    tol : float, optional
        Gradient norm tolerance. Default: ``torch.finfo(x0.dtype).eps ** 0.5``.
    max_cg_iter : int, optional
        Maximum CG iterations per trust-region subproblem. Default: 50.
    initial_trust_radius : float, optional
        Initial trust-region radius. Default: 1.0.
    max_trust_radius : float, optional
        Maximum trust-region radius. Default: 100.0.
    eta : float, optional
        Acceptance threshold for step. Default: 0.15.

    Returns
    -------
    OptimizeResult
        Result with fields ``x``, ``converged``, ``num_iterations``,
        ``fun``.
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    x = x0.clone().detach()
    f_val, g = _compute_grad(fun, x, grad)
    trust_radius = initial_trust_radius
    num_iter = 0
    converged = torch.tensor(False, device=x.device)

    for k in range(maxiter):
        grad_inf = g.abs().max()
        if grad_inf < tol:
            converged = torch.tensor(True, device=x.device)
            break

        num_iter = k + 1

        # Solve trust-region subproblem
        def hvp_fn(v):
            return _hessian_vector_product(fun, x, v)

        step = _steihaug_cg(hvp_fn, g, trust_radius, max_cg_iter)

        # Predicted reduction: -g^T d - 0.5 d^T H d
        Hd = hvp_fn(step)
        predicted = -(g * step).sum() - 0.5 * (step * Hd).sum()

        # Actual reduction
        x_new = x + step
        f_new, g_new = _compute_grad(fun, x_new, grad)
        actual = f_val - f_new

        # Ratio
        if predicted.item() > 0:
            rho = (actual / predicted).item()
        else:
            rho = 0.0

        # Update trust radius
        step_norm = step.norm().item()
        if rho < 0.25:
            trust_radius = 0.25 * step_norm
        elif rho > 0.75 and abs(step_norm - trust_radius) < 1e-10:
            trust_radius = min(2.0 * trust_radius, max_trust_radius)

        # Prevent trust radius from collapsing to zero
        if trust_radius < 1e-15:
            trust_radius = initial_trust_radius * 0.01

        # Accept or reject step
        if rho > eta:
            x = x_new
            g = g_new
            f_val = f_new
    else:
        converged = torch.tensor(g.abs().max() < tol, device=x.device)

    with torch.no_grad():
        f_final = fun(x)

    x_with_grad = _implicit_diff_step(fun, x)

    return OptimizeResult(
        x=x_with_grad,
        converged=converged,
        num_iterations=torch.tensor(
            num_iter, dtype=torch.int64, device=x.device
        ),
        fun=f_final,
    )

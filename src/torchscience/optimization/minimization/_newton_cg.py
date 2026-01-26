from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._line_search import _strong_wolfe_line_search
from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._l_bfgs import (
    _compute_grad,
    _implicit_diff_step,
)


def _hessian_vector_product(fun, x, v):
    """Compute H @ v via reverse-mode AD: d/dx [g(x)^T v].

    Computes the Hessian-vector product without forming the full Hessian.
    """
    x_hvp = x.detach().requires_grad_(True)
    f_val = fun(x_hvp)
    if f_val.dim() > 0:
        f_val = f_val.sum()
    g = torch.autograd.grad(f_val, x_hvp, create_graph=True)[0]
    gv = (g * v.detach()).sum()
    hvp = torch.autograd.grad(gv, x_hvp)[0]
    return hvp.detach()


def _cg_steihaug(hvp_fn, grad, tol_cg, max_cg_iter):
    """Solve H @ d = -grad approximately using CG with negative curvature check."""
    n = grad.numel()
    d = torch.zeros_like(grad)
    r = grad.clone()
    p = -r.clone()

    r_dot_r = (r * r).sum()
    tol_sq = tol_cg**2

    for _ in range(min(max_cg_iter, n)):
        if r_dot_r < tol_sq:
            break

        Hp = hvp_fn(p)
        pHp = (p * Hp).sum()

        # Negative curvature: return current d (Steihaug truncation)
        if pHp <= 0:
            if d.norm() == 0:
                return -grad  # Steepest descent fallback
            return d

        alpha = r_dot_r / pHp
        d = d + alpha * p
        r = r + alpha * Hp

        r_dot_r_new = (r * r).sum()
        beta = r_dot_r_new / r_dot_r.clamp(min=1e-30)
        p = -r + beta * p
        r_dot_r = r_dot_r_new

    return d


def newton_cg(
    fun: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    grad: Optional[Callable[[Tensor], Tensor]] = None,
    maxiter: int = 100,
    tol: Optional[float] = None,
    max_cg_iter: int = 50,
    line_search: str = "strong_wolfe",
) -> OptimizeResult:
    r"""
    Newton-CG (truncated Newton) method for unconstrained optimization.

    Uses conjugate gradient to approximately solve the Newton system
    ``H d = -g`` at each iteration, where ``H`` is the Hessian and ``g``
    is the gradient. Hessian-vector products are computed via
    ``torch.func.jvp`` without forming the full Hessian.

    Parameters
    ----------
    fun : Callable[[Tensor], Tensor]
        Scalar-valued objective function to minimize.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    grad : Callable[[Tensor], Tensor], optional
        Gradient function. If ``None``, uses ``torch.autograd``.
    maxiter : int, optional
        Maximum number of outer Newton iterations. Default: 100.
    tol : float, optional
        Gradient norm tolerance for convergence. Default:
        ``torch.finfo(x0.dtype).eps ** 0.5``.
    max_cg_iter : int, optional
        Maximum CG iterations per Newton step. Default: 50.
    line_search : str, optional
        Line search method. Default: ``"strong_wolfe"``.

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
    num_iter = 0

    def _f_and_grad(x_inner):
        x_inner = x_inner.detach().requires_grad_(True)
        val = fun(x_inner)
        if val.dim() > 0:
            val_scalar = val.sum()
        else:
            val_scalar = val
        g_inner = torch.autograd.grad(val_scalar, x_inner)[0]
        return val.detach(), g_inner.detach()

    for k in range(maxiter):
        grad_inf = g.abs().max()
        if grad_inf < tol:
            converged = torch.tensor(True, device=x.device)
            break

        num_iter = k + 1

        # CG tolerance decreases with gradient norm (Eisenstat-Walker)
        tol_cg = min(0.5, g.norm().item() ** 0.5) * g.norm().item()

        # Solve Newton system approximately
        def hvp_fn(v):
            return _hessian_vector_product(fun, x, v)

        direction = _cg_steihaug(hvp_fn, g, tol_cg, max_cg_iter)

        # Ensure descent direction
        dg = (g * direction).sum()
        if dg >= 0:
            direction = -g

        # Line search
        if line_search == "strong_wolfe":
            alpha = _strong_wolfe_line_search(
                _f_and_grad,
                x,
                direction,
                f_val,
                g,
            )
        else:
            alpha = torch.tensor(1.0, dtype=x.dtype, device=x.device)

        x_new = x + alpha * direction
        f_new, g_new = _compute_grad(fun, x_new, grad)

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

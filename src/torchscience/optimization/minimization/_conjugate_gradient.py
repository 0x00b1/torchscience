from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._line_search import _strong_wolfe_line_search
from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._l_bfgs import (
    _compute_grad,
    _implicit_diff_step,
)


def conjugate_gradient(
    fun: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    grad: Optional[Callable[[Tensor], Tensor]] = None,
    maxiter: int = 200,
    tol: Optional[float] = None,
    variant: str = "polak-ribiere+",
    line_search: str = "strong_wolfe",
) -> OptimizeResult:
    r"""
    Nonlinear conjugate gradient method for unconstrained optimization.

    Minimizes a scalar-valued function using the nonlinear conjugate
    gradient method. Supports multiple update formulas: Polak-Ribiere+
    (default, with automatic restart), Fletcher-Reeves, and
    Hestenes-Stiefel.

    Parameters
    ----------
    fun : Callable[[Tensor], Tensor]
        Scalar-valued objective function to minimize.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    grad : Callable[[Tensor], Tensor], optional
        Gradient function. If ``None``, uses ``torch.autograd``.
    maxiter : int, optional
        Maximum number of iterations. Default: 200.
    tol : float, optional
        Gradient norm tolerance for convergence. Default:
        ``torch.finfo(x0.dtype).eps ** 0.5``.
    variant : str, optional
        CG update formula. One of ``"polak-ribiere+"`` (default),
        ``"fletcher-reeves"``, or ``"hestenes-stiefel"``.
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

    variant_lower = variant.lower().replace("_", "-")

    x = x0.clone().detach()
    f_val, g = _compute_grad(fun, x, grad)

    d = -g.clone()
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

        # Line search
        dg = (g * d).sum()
        if dg >= 0:
            # Direction is not a descent direction; restart
            d = -g.clone()
            dg = (g * d).sum()

        if line_search == "strong_wolfe":
            alpha = _strong_wolfe_line_search(
                _f_and_grad,
                x,
                d,
                f_val,
                g,
            )
        else:
            alpha = torch.tensor(1.0, dtype=x.dtype, device=x.device)

        x_new = x + alpha * d
        f_new, g_new = _compute_grad(fun, x_new, grad)

        # Compute beta
        if variant_lower == "fletcher-reeves":
            beta = (g_new * g_new).sum() / (g * g).sum().clamp(min=1e-30)
        elif variant_lower == "hestenes-stiefel":
            y = g_new - g
            beta = (g_new * y).sum() / (d * y).sum().clamp(min=1e-30)
        else:
            # Polak-Ribiere+ (default)
            y = g_new - g
            beta = (g_new * y).sum() / (g * g).sum().clamp(min=1e-30)
            beta = torch.clamp(beta, min=0.0)  # PR+ restart

        d = -g_new + beta * d

        x = x_new
        g = g_new
        f_val = f_new
    else:
        converged = (g.abs().max() < tol).clone()

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

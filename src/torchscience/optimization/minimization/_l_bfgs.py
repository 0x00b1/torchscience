"""L-BFGS solver for unconstrained optimization."""

from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._line_search import (
    _backtracking_line_search,
    _strong_wolfe_line_search,
)
from torchscience.optimization._result import OptimizeResult


def _two_loop_recursion(
    grad: Tensor,
    s_history: list[Tensor],
    y_history: list[Tensor],
    rho_history: list[Tensor],
) -> Tensor:
    """L-BFGS two-loop recursion (Nocedal & Wright, Algorithm 7.4).

    Computes the search direction ``H_k @ grad`` where ``H_k`` is the
    inverse Hessian approximation built from the stored correction pairs.

    Parameters
    ----------
    grad : Tensor
        Current gradient of shape ``(..., n)``.
    s_history : list[Tensor]
        List of ``s_i = x_{i+1} - x_i`` vectors.
    y_history : list[Tensor]
        List of ``y_i = grad_{i+1} - grad_i`` vectors.
    rho_history : list[Tensor]
        List of ``1 / (y_i^T s_i)`` scalars.

    Returns
    -------
    Tensor
        Search direction of shape ``(..., n)``.
    """
    q = grad.clone()
    m = len(s_history)

    alphas = []
    for i in range(m - 1, -1, -1):
        alpha = rho_history[i] * (s_history[i] * q).sum(dim=-1, keepdim=True)
        q = q - alpha * y_history[i]
        alphas.append(alpha)
    alphas.reverse()

    # Initial Hessian scaling: gamma = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
    if m > 0:
        gamma = (s_history[-1] * y_history[-1]).sum(dim=-1, keepdim=True) / (
            y_history[-1] * y_history[-1]
        ).sum(dim=-1, keepdim=True).clamp(min=1e-30)
        r = gamma * q
    else:
        r = q

    for i in range(m):
        beta = rho_history[i] * (y_history[i] * r).sum(dim=-1, keepdim=True)
        r = r + s_history[i] * (alphas[i] - beta)

    return r


def _compute_grad(
    fun: Callable[[Tensor], Tensor],
    x: Tensor,
    grad_fn: Optional[Callable[[Tensor], Tensor]] = None,
) -> tuple[Tensor, Tensor]:
    """Compute function value and gradient.

    Parameters
    ----------
    fun : Callable
        Objective function.
    x : Tensor
        Current point.
    grad_fn : Callable, optional
        Explicit gradient function. If ``None``, uses ``torch.autograd``.

    Returns
    -------
    tuple[Tensor, Tensor]
        Function value and gradient.
    """
    if grad_fn is not None:
        with torch.no_grad():
            f_val = fun(x)
        g = grad_fn(x)
        return f_val.detach(), g.detach()

    x_grad = x.detach().requires_grad_(True)
    f_val = fun(x_grad)

    # Handle batched case: sum to get scalar for autograd
    if f_val.dim() > 0:
        f_scalar = f_val.sum()
    else:
        f_scalar = f_val

    g = torch.autograd.grad(f_scalar, x_grad)[0]
    return f_val.detach(), g.detach()


def _implicit_diff_step(
    fun: Callable[[Tensor], Tensor],
    x_opt: Tensor,
) -> Tensor:
    """Attach implicit gradient to the optimum via one Newton correction.

    At the optimum ``x*``, ``grad_x f(x*; theta) approx 0``. By the implicit
    function theorem:

    .. math::

        \\frac{dx^*}{d\\theta} = -[\\nabla^2_{xx} f]^{-1} \\nabla^2_{x\\theta} f

    This is implemented by computing:

    .. math::

        x_{\\text{result}} = x^* - H^{-1} \\nabla_x f(x^*)

    where ``H`` is the (detached) Hessian. Since ``grad_x f approx 0`` at
    convergence, ``x_result approx x*`` in value, but carries the correct
    implicit gradient through the ``grad_x f`` term.

    Parameters
    ----------
    fun : Callable
        Objective function (captures external parameters ``theta``).
    x_opt : Tensor
        Detached optimum of shape ``(n,)``.

    Returns
    -------
    Tensor
        Tensor equal to ``x_opt`` in value but with implicit gradient attached.
    """
    with torch.enable_grad():
        x_for_grad = x_opt.detach().requires_grad_(True)
        f_val = fun(x_for_grad)

        if f_val.dim() > 0:
            f_scalar = f_val.sum()
        else:
            f_scalar = f_val

        grad_at_opt = torch.autograd.grad(
            f_scalar,
            x_for_grad,
            create_graph=True,
        )[0]

        # Compute Hessian (detached) for the linear solve
        n = x_for_grad.numel()
        hess_rows = []
        flat_grad = grad_at_opt.flatten()
        for i in range(n):
            h_row = torch.autograd.grad(
                flat_grad[i],
                x_for_grad,
                retain_graph=True,
                create_graph=False,
            )[0]
            if h_row is None:
                h_row = torch.zeros_like(x_for_grad)
            hess_rows.append(h_row.flatten())
        H = torch.stack(hess_rows).detach()

        reg = 1e-6 * torch.eye(n, dtype=H.dtype, device=H.device)
        # Solve H @ correction = grad_at_opt
        # grad_at_opt retains grad_fn → captured params theta
        correction = torch.linalg.solve(
            H + reg,
            grad_at_opt.flatten(),
        )
        x_result = x_opt.detach() - correction.reshape(x_opt.shape)

    return x_result


def l_bfgs(
    fun: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    grad: Optional[Callable[[Tensor], Tensor]] = None,
    maxiter: int = 100,
    tol: Optional[float] = None,
    history_size: int = 10,
    line_search: str = "strong_wolfe",
) -> OptimizeResult:
    r"""L-BFGS algorithm for unconstrained optimization.

    Finds parameters ``x`` that minimize:

    .. math::

        \min_x f(x)

    The limited-memory BFGS algorithm approximates the inverse Hessian
    using the most recent correction pairs, requiring only ``O(n \times m)``
    storage where ``m`` is the history size.

    Parameters
    ----------
    fun : Callable[[Tensor], Tensor]
        Objective function. Takes a tensor of shape ``(..., n)`` and returns
        a scalar (unbatched) or tensor of shape ``(...,)`` (batched).
    x0 : Tensor
        Initial guess of shape ``(n,)`` or ``(..., n)``.
    grad : Callable, optional
        Gradient function. If ``None``, computed via ``torch.autograd``.
        Should return a tensor of the same shape as the input.
    maxiter : int
        Maximum number of iterations. Default: 100.
    tol : float, optional
        Convergence tolerance on gradient infinity norm.
        Default: ``sqrt(eps)`` for the dtype.
    history_size : int
        Number of correction pairs to store. Default: 10.
    line_search : str
        Line search method: ``"strong_wolfe"`` (default) or ``"armijo"``.

    Returns
    -------
    OptimizeResult
        Named tuple with fields:

        - ``x``: Solution tensor of the same shape as ``x0``.
        - ``converged``: Boolean convergence indicator.
        - ``num_iterations``: Number of iterations performed.
        - ``fun``: Objective value at the solution.

    Examples
    --------
    Minimize a quadratic:

    >>> def f(x):
    ...     return (x ** 2).sum()
    >>> result = l_bfgs(f, torch.tensor([3.0, 4.0]))
    >>> result.x
    tensor([0., 0.])

    Minimize the Rosenbrock function:

    >>> def rosenbrock(x):
    ...     return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    >>> result = l_bfgs(rosenbrock, torch.tensor([-1.0, 1.0]))
    >>> result.converged
    tensor(True)

    The optimizer supports implicit differentiation:

    >>> target = torch.tensor([5.0], requires_grad=True)
    >>> def f(x):
    ...     return ((x - target) ** 2).sum()
    >>> result = l_bfgs(f, torch.zeros(1))
    >>> result.x.sum().backward()
    >>> target.grad
    tensor([1.])

    References
    ----------
    - Nocedal, J. "Updating quasi-Newton matrices with limited storage."
      Mathematics of computation 35.151 (1980): 773-782.
    - Liu, D.C. and Nocedal, J. "On the limited memory BFGS method for
      large scale optimization." Mathematical programming 45.1 (1989): 503-528.
    - Nocedal, J. and Wright, S.J. "Numerical Optimization." Algorithm 7.4.

    See Also
    --------
    https://en.wikipedia.org/wiki/Limited-memory_BFGS
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    batched = x0.dim() > 1
    x = x0.detach().clone()

    s_history: list[Tensor] = []
    y_history: list[Tensor] = []
    rho_history: list[Tensor] = []

    f_val, g = _compute_grad(fun, x, grad)

    converged = torch.zeros(
        x.shape[:-1] if batched else (),
        dtype=torch.bool,
        device=x.device,
    )
    num_iter = 0

    for k in range(maxiter):
        # Check convergence: ||grad||_inf < tol
        if batched:
            grad_inf = g.abs().amax(dim=-1)
            converged = grad_inf < tol
            if converged.all():
                break
        else:
            grad_inf = g.abs().max()
            if grad_inf < tol:
                converged = torch.tensor(True, device=x.device)
                break

        num_iter = k + 1

        # Compute search direction via two-loop recursion
        direction = -_two_loop_recursion(g, s_history, y_history, rho_history)

        # Line search
        if line_search == "strong_wolfe" and not batched:

            def f_and_grad_fn(x_new: Tensor) -> tuple[Tensor, Tensor]:
                return _compute_grad(fun, x_new, grad)

            dg = torch.dot(g.flatten(), direction.flatten())
            if dg >= 0:
                # Reset direction to steepest descent if not a descent direction
                direction = -g
                dg = torch.dot(g.flatten(), direction.flatten())

            alpha = _strong_wolfe_line_search(
                f_and_grad_fn,
                x,
                direction,
                f_val,
                g,
            )
        else:
            # Armijo backtracking (supports batched)
            if batched:
                grad_dot_dir = (g * direction).sum(dim=-1)
                # Ensure descent direction
                not_descent = grad_dot_dir >= 0
                if not_descent.any():
                    direction = torch.where(
                        not_descent.unsqueeze(-1),
                        -g,
                        direction,
                    )
                    grad_dot_dir = (g * direction).sum(dim=-1)
            else:
                grad_dot_dir = torch.dot(g.flatten(), direction.flatten())
                if grad_dot_dir >= 0:
                    direction = -g
                    grad_dot_dir = torch.dot(g.flatten(), direction.flatten())

            def f_eval(x_new: Tensor) -> Tensor:
                with torch.no_grad():
                    val = fun(x_new)
                return val

            alpha = _backtracking_line_search(
                f_eval,
                x,
                direction,
                grad_dot_dir,
                f_val,
            )

        # Update x
        if batched:
            alpha_expanded = alpha.unsqueeze(-1)
        else:
            alpha_expanded = alpha

        x_new = x + alpha_expanded * direction
        f_new, g_new = _compute_grad(fun, x_new, grad)

        # Update history
        s = x_new - x
        y = g_new - g
        ys = (y * s).sum(dim=-1, keepdim=True)

        # Only update if curvature condition is satisfied
        if batched:
            valid = ys.squeeze(-1) > 1e-10
            if valid.any():
                rho_val = torch.where(
                    valid.unsqueeze(-1),
                    1.0 / ys.clamp(min=1e-30),
                    torch.zeros_like(ys),
                )
                s_history.append(s)
                y_history.append(y)
                rho_history.append(rho_val)
        else:
            if ys.item() > 1e-10:
                rho_val = 1.0 / ys
                s_history.append(s)
                y_history.append(y)
                rho_history.append(rho_val)

        # Enforce history size limit
        if len(s_history) > history_size:
            s_history.pop(0)
            y_history.pop(0)
            rho_history.pop(0)

        x = x_new
        g = g_new
        f_val = f_new

    # Final convergence check
    if batched:
        grad_inf = g.abs().amax(dim=-1)
        converged = grad_inf < tol
    else:
        converged = (g.abs().max() < tol).clone()

    # Evaluate function at solution
    with torch.no_grad():
        f_final = fun(x)

    # Attach implicit gradient for backpropagation
    if batched:
        x_with_grad = x
    else:
        x_with_grad = _implicit_diff_step(fun, x)

    return OptimizeResult(
        x=x_with_grad,
        converged=converged,
        num_iterations=torch.tensor(
            num_iter, dtype=torch.int64, device=x.device
        ),
        fun=f_final,
    )

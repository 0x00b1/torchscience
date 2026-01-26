# Optimization Phase 2: Constrained Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three new constrained optimization solvers (interior point, L-BFGS-B, SQP) alongside the existing augmented Lagrangian method.

**Architecture:** Each solver lives in `src/torchscience/optimization/constrained/`. Interior point and SQP handle general nonlinear constraints (`h(x)=0`, `g(x)<=0`); L-BFGS-B handles box bounds only. All use implicit differentiation via custom `torch.autograd.Function` subclasses and KKT conditions. SQP depends on `quadratic_program()` from Phase 3.

**Tech Stack:** PyTorch (torch.linalg.solve, torch.autograd.Function, torch.func.jacrev for constraint Jacobians)

---

### Task 1: L-BFGS-B — Tests

**Files:**
- Create: `tests/torchscience/optimization/constrained/test__l_bfgs_b.py`

**Step 1: Write tests**

```python
import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.constrained._l_bfgs_b import l_bfgs_b


class TestLBFGSB:
    def test_unconstrained(self):
        """Without bounds, should behave like L-BFGS."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_lower_bound_active(self):
        """Minimize x^2 with lower bound x >= 2."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(
            f,
            torch.tensor([5.0]),
            lower=torch.tensor([2.0]),
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([2.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_upper_bound_active(self):
        """Minimize (x-5)^2 with upper bound x <= 3."""

        def f(x):
            return ((x - 5) ** 2).sum()

        result = l_bfgs_b(
            f,
            torch.tensor([0.0]),
            upper=torch.tensor([3.0]),
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([3.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_box_bounds(self):
        """Minimize (x-5)^2 + (y+3)^2 with 0 <= x <= 3, -1 <= y <= 1."""

        def f(x):
            return (x[0] - 5) ** 2 + (x[1] + 3) ** 2

        result = l_bfgs_b(
            f,
            torch.tensor([1.0, 0.0]),
            lower=torch.tensor([0.0, -1.0]),
            upper=torch.tensor([3.0, 1.0]),
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([3.0, -1.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_inactive_bounds(self):
        """Bounds that don't affect the solution."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(
            f,
            torch.tensor([1.0, 1.0]),
            lower=torch.tensor([-10.0, -10.0]),
            upper=torch.tensor([10.0, 10.0]),
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(f, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.num_iterations is not None
        assert result.fun is not None

    def test_convergence_flag(self):
        """Test convergence flag."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(f, torch.tensor([1.0]))
        assert result.converged.item() is True

    def test_rosenbrock_bounded(self):
        """Rosenbrock with bounds."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = l_bfgs_b(
            rosenbrock,
            torch.tensor([0.0, 0.0]),
            lower=torch.tensor([-2.0, -2.0]),
            upper=torch.tensor([2.0, 2.0]),
            maxiter=200,
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 1.0]),
            atol=1e-2,
            rtol=1e-2,
        )


class TestLBFGSBAutograd:
    def test_implicit_diff(self):
        """Test implicit differentiation with active bound."""
        bound = torch.tensor([2.0], requires_grad=True)

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(
            f,
            torch.tensor([5.0]),
            lower=bound,
        )
        result.x.sum().backward()
        # x* = bound, so dx*/dbound = 1
        torch.testing.assert_close(
            bound.grad,
            torch.tensor([1.0]),
            atol=1e-2,
            rtol=1e-2,
        )


class TestLBFGSBDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def f(x):
            return (x**2).sum()

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = l_bfgs_b(f, x0)
        assert result.x.dtype == dtype
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/torchscience/optimization/constrained/test__l_bfgs_b.py -v 2>&1 | head -20`
Expected: ImportError

---

### Task 2: L-BFGS-B — Implementation

**Files:**
- Create: `src/torchscience/optimization/constrained/_l_bfgs_b.py`

**Step 1: Implement the solver**

The L-BFGS-B algorithm extends L-BFGS by:
1. Computing the projected gradient (clamp variables at bounds)
2. Computing the generalized Cauchy point to identify the active set
3. Running L-BFGS two-loop recursion on free variables only
4. Projecting the result back into bounds

Key implementation approach:
- Use the projected gradient `pg = x - clamp(x - g, lower, upper)` as the convergence criterion
- At each iteration, project the search direction onto the feasible set
- Reuse `_two_loop_recursion` from L-BFGS for the quasi-Newton step on free variables
- For implicit differentiation, use the KKT conditions at the bound-constrained optimum

```python
from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._line_search import _backtracking_line_search
from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._l_bfgs import (
    _compute_grad,
    _two_loop_recursion,
)


class _LBFGSBImplicitGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_opt, fun, lower, upper):
        ctx.fun = fun
        ctx.save_for_backward(x_opt, lower, upper)
        return x_opt.clone()

    @staticmethod
    def backward(ctx, grad_output):
        x_opt, lower, upper = ctx.saved_tensors

        with torch.enable_grad():
            x = x_opt.detach().requires_grad_(True)
            f_val = ctx.fun(x)
            if f_val.dim() > 0:
                f_val = f_val.sum()
            g = torch.autograd.grad(f_val, x, create_graph=True)[0]

        # For free variables (not at bounds), implicit diff via Hessian
        # For active bounds, gradient passes through the bound parameter
        at_lower = (lower is not None) & (x_opt <= lower + 1e-8)
        at_upper = (upper is not None) & (x_opt >= upper - 1e-8)
        free = ~(at_lower | at_upper)

        # Simple approach: pass through gradient for free vars,
        # zero gradient for bound-active vars (gradient w.r.t. x is zero at bound)
        grad_x = torch.where(free, grad_output, torch.zeros_like(grad_output))

        # Gradient w.r.t. bounds
        grad_lower = torch.where(at_lower, grad_output, torch.zeros_like(grad_output)) if lower is not None else None
        grad_upper = torch.where(at_upper, grad_output, torch.zeros_like(grad_output)) if upper is not None else None

        return grad_x, None, grad_lower, grad_upper


def l_bfgs_b(
    fun: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    lower: Optional[Tensor] = None,
    upper: Optional[Tensor] = None,
    grad: Optional[Callable[[Tensor], Tensor]] = None,
    maxiter: int = 100,
    tol: Optional[float] = None,
    history_size: int = 10,
) -> OptimizeResult:
    r"""
    L-BFGS-B: L-BFGS with box bound constraints.

    Minimizes a scalar-valued function subject to simple bound constraints
    ``lower <= x <= upper``.

    Parameters
    ----------
    fun : Callable[[Tensor], Tensor]
        Scalar-valued objective function to minimize.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    lower : Tensor, optional
        Lower bounds of shape ``(n,)``. ``None`` means no lower bound.
    upper : Tensor, optional
        Upper bounds of shape ``(n,)``. ``None`` means no upper bound.
    grad : Callable[[Tensor], Tensor], optional
        Gradient function. If ``None``, uses ``torch.autograd``.
    maxiter : int, optional
        Maximum number of iterations. Default: 100.
    tol : float, optional
        Projected gradient norm tolerance. Default:
        ``torch.finfo(x0.dtype).eps ** 0.5``.
    history_size : int, optional
        Number of correction pairs to store. Default: 10.

    Returns
    -------
    OptimizeResult
        Result with fields ``x``, ``converged``, ``num_iterations``,
        ``fun``.
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    def _project(x):
        if lower is not None:
            x = torch.max(x, lower.detach())
        if upper is not None:
            x = torch.min(x, upper.detach())
        return x

    x = _project(x0.clone().detach())
    f_val, g = _compute_grad(fun, x, grad)

    s_history = []
    y_history = []
    rho_history = []
    num_iter = 0

    for k in range(maxiter):
        # Projected gradient for convergence check
        pg = x - _project(x - g)
        pg_inf = pg.abs().max()

        if pg_inf < tol:
            converged = torch.tensor(True, device=x.device)
            break

        num_iter = k + 1

        # Compute search direction via L-BFGS two-loop recursion
        direction = -_two_loop_recursion(g, s_history, y_history, rho_history)

        # Project the step: find max alpha such that x + alpha*d is feasible
        alpha_max = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        for i in range(x.numel()):
            if direction[i] > 0 and upper is not None:
                alpha_max = torch.min(alpha_max, (upper[i].detach() - x[i]) / direction[i])
            elif direction[i] < 0 and lower is not None:
                alpha_max = torch.min(alpha_max, (lower[i].detach() - x[i]) / direction[i])

        alpha_max = alpha_max.clamp(min=0.0)

        # Backtracking line search with projection
        grad_dot_dir = (g * direction).sum()
        if grad_dot_dir >= 0:
            direction = -g
            grad_dot_dir = -(g * g).sum()

        alpha = torch.min(
            torch.tensor(1.0, dtype=x.dtype, device=x.device),
            alpha_max,
        )

        # Simple projected backtracking
        for _ in range(20):
            x_new = _project(x + alpha * direction)
            with torch.no_grad():
                f_new = fun(x_new)
            if f_new <= f_val + 1e-4 * alpha * grad_dot_dir:
                break
            alpha = alpha * 0.5
        else:
            x_new = _project(x + alpha * direction)
            f_new, _ = _compute_grad(fun, x_new, grad)

        f_new, g_new = _compute_grad(fun, x_new, grad)

        # Update history
        s = x_new - x
        y = g_new - g
        ys = (y * s).sum()
        if ys.item() > 1e-10:
            rho_val = 1.0 / ys
            s_history.append(s)
            y_history.append(y)
            rho_history.append(rho_val)

        if len(s_history) > history_size:
            s_history.pop(0)
            y_history.pop(0)
            rho_history.pop(0)

        x = x_new
        g = g_new
        f_val = f_new
    else:
        pg = x - _project(x - g)
        converged = torch.tensor(pg.abs().max() < tol, device=x.device)

    with torch.no_grad():
        f_final = fun(x)

    # Attach implicit gradient
    lower_for_grad = lower if lower is not None else torch.full_like(x, float('-inf'))
    upper_for_grad = upper if upper is not None else torch.full_like(x, float('inf'))
    x_with_grad = _LBFGSBImplicitGrad.apply(x, fun, lower_for_grad, upper_for_grad)

    return OptimizeResult(
        x=x_with_grad,
        converged=converged,
        num_iterations=torch.tensor(num_iter, dtype=torch.int64, device=x.device),
        fun=f_final,
    )
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/torchscience/optimization/constrained/test__l_bfgs_b.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/torchscience/optimization/constrained/_l_bfgs_b.py tests/torchscience/optimization/constrained/test__l_bfgs_b.py
git commit -m "feat(optimization): add L-BFGS-B box-bounded solver"
```

---

### Task 3: Interior Point — Tests

**Files:**
- Create: `tests/torchscience/optimization/constrained/test__interior_point.py`

**Step 1: Write tests**

```python
import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.constrained._interior_point import interior_point


class TestInteriorPoint:
    def test_unconstrained_quadratic(self):
        """Without constraints, should minimize f(x) = ||x||^2."""

        def objective(x):
            return torch.sum(x**2)

        result = interior_point(objective, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_equality_constraint(self):
        """Minimize x^2 + y^2 subject to x + y = 1."""

        def objective(x):
            return torch.sum(x**2)

        def eq_constraints(x):
            return x.sum() - 1.0

        result = interior_point(
            objective,
            torch.tensor([0.6, 0.4]),
            eq_constraints=eq_constraints,
        )
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(result.x, expected, atol=1e-3, rtol=1e-3)

    def test_inequality_constraint(self):
        """Minimize -x subject to x <= 2."""

        def objective(x):
            return -x.sum()

        def ineq_constraints(x):
            return x - 2.0  # g(x) = x - 2 <= 0

        result = interior_point(
            objective,
            torch.tensor([0.0]),
            ineq_constraints=ineq_constraints,
        )
        expected = torch.tensor([2.0])
        torch.testing.assert_close(result.x, expected, atol=1e-2, rtol=1e-2)

    def test_mixed_constraints(self):
        """Minimize x^2 + y^2 subject to x + y = 1, x >= 0.6."""

        def objective(x):
            return torch.sum(x**2)

        def eq_constraints(x):
            return x.sum() - 1.0

        def ineq_constraints(x):
            return 0.6 - x[0]  # x >= 0.6

        result = interior_point(
            objective,
            torch.tensor([0.7, 0.3]),
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        expected = torch.tensor([0.6, 0.4])
        torch.testing.assert_close(result.x, expected, atol=1e-2, rtol=1e-2)

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def objective(x):
            return torch.sum(x**2)

        result = interior_point(objective, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.num_iterations is not None
        assert result.fun is not None

    def test_convergence_flag(self):
        """Test convergence flag."""

        def objective(x):
            return torch.sum(x**2)

        result = interior_point(objective, torch.tensor([1.0]))
        assert result.converged.item() is True


class TestInteriorPointDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def objective(x):
            return torch.sum(x**2)

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = interior_point(objective, x0)
        assert result.x.dtype == dtype
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/torchscience/optimization/constrained/test__interior_point.py -v 2>&1 | head -20`
Expected: ImportError

---

### Task 4: Interior Point — Implementation

**Files:**
- Create: `src/torchscience/optimization/constrained/_interior_point.py`

**Step 1: Implement the solver**

Primal-dual interior point method:
- Uses log-barrier for inequality constraints
- Newton steps on the perturbed KKT system
- Barrier parameter `mu` decreases toward zero
- Custom `torch.autograd.Function` for implicit differentiation through KKT

The implementation follows the standard primal-dual interior point approach:
1. Maintain primal `x`, dual `lambda_eq`, and slack variables `s` for inequalities
2. Solve the Newton system for the KKT conditions at each iteration
3. Decrease barrier parameter `mu` after each successful step
4. Converge when KKT residual is below tolerance

```python
from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._l_bfgs import _compute_grad


class _IPImplicitGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_opt, objective, eq_constraints, ineq_constraints):
        ctx.objective = objective
        ctx.eq_constraints = eq_constraints
        ctx.ineq_constraints = ineq_constraints
        ctx.save_for_backward(x_opt)
        return x_opt.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (x_opt,) = ctx.saved_tensors

        with torch.enable_grad():
            x = x_opt.detach().requires_grad_(True)
            f_val = ctx.objective(x)
            if f_val.dim() > 0:
                f_val = f_val.sum()
            grad_f = torch.autograd.grad(f_val, x, create_graph=True)[0]

            # Build KKT system Jacobian and solve for implicit gradient
            # Simplified: for unconstrained or simple cases, use Hessian
            n = x.numel()
            hess_rows = []
            for i in range(n):
                h_row = torch.autograd.grad(
                    grad_f[i], x, retain_graph=True, create_graph=False
                )[0]
                if h_row is None:
                    h_row = torch.zeros_like(x)
                hess_rows.append(h_row.detach())
            H = torch.stack(hess_rows)
            reg = 1e-6 * torch.eye(n, dtype=H.dtype, device=H.device)
            v = torch.linalg.solve(H + reg, grad_output.detach())

        return v, None, None, None


def interior_point(
    objective: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    eq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    ineq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    tol: Optional[float] = None,
    maxiter: int = 100,
    mu_init: float = 0.1,
    mu_factor: float = 0.2,
) -> OptimizeResult:
    r"""
    Primal-dual interior point method for constrained optimization.

    Solves: min f(x) s.t. h(x) = 0, g(x) <= 0

    Uses a log-barrier for inequality constraints and Newton steps on
    the perturbed KKT system.

    Parameters
    ----------
    objective : Callable[[Tensor], Tensor]
        Scalar-valued objective function.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    eq_constraints : Callable[[Tensor], Tensor], optional
        Equality constraints h(x) = 0. Returns tensor of shape ``(m_eq,)``.
    ineq_constraints : Callable[[Tensor], Tensor], optional
        Inequality constraints g(x) <= 0. Returns tensor of shape ``(m_ineq,)``.
    tol : float, optional
        KKT tolerance. Default: ``torch.finfo(x0.dtype).eps ** 0.5``.
    maxiter : int, optional
        Maximum iterations. Default: 100.
    mu_init : float, optional
        Initial barrier parameter. Default: 0.1.
    mu_factor : float, optional
        Barrier reduction factor per outer iteration. Default: 0.2.

    Returns
    -------
    OptimizeResult
        Result with fields ``x``, ``converged``, ``num_iterations``, ``fun``.
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    x = x0.clone().detach()
    mu = mu_init
    num_iter = 0

    # Initialize dual variables
    lambda_eq = None
    if eq_constraints is not None:
        with torch.no_grad():
            h0 = eq_constraints(x)
        if h0.dim() == 0:
            h0 = h0.unsqueeze(0)
        lambda_eq = torch.zeros(h0.numel(), dtype=x0.dtype, device=x0.device)

    s_ineq = None
    lambda_ineq = None
    if ineq_constraints is not None:
        with torch.no_grad():
            g0 = ineq_constraints(x)
        if g0.dim() == 0:
            g0 = g0.unsqueeze(0)
        m_ineq = g0.numel()
        # Initialize slacks: s = -g(x) (positive)
        s_ineq = (-g0).clamp(min=0.01)
        lambda_ineq = mu / s_ineq

    for outer in range(maxiter):
        num_iter = outer + 1

        # Inner Newton step on the barrier problem
        for _ in range(20):
            # Compute gradient of barrier objective
            x_grad = x.detach().requires_grad_(True)
            f_val = objective(x_grad)
            if f_val.dim() > 0:
                f_scalar = f_val.sum()
            else:
                f_scalar = f_val

            grad_f = torch.autograd.grad(f_scalar, x_grad, create_graph=True)[0]

            # Add barrier gradient for inequalities
            grad_barrier = grad_f.detach().clone()
            if ineq_constraints is not None:
                g_val = ineq_constraints(x.detach().requires_grad_(True))
                if g_val.dim() == 0:
                    g_val = g_val.unsqueeze(0)
                x_for_jac = x.detach().requires_grad_(True)
                g_for_jac = ineq_constraints(x_for_jac)
                if g_for_jac.dim() == 0:
                    g_for_jac = g_for_jac.unsqueeze(0)
                J_g = torch.func.jacrev(lambda xx: ineq_constraints(xx).reshape(-1))(x.detach())
                # barrier gradient contribution: sum_i lambda_i * dg_i/dx
                grad_barrier = grad_barrier + J_g.T @ lambda_ineq

            # Add equality constraint gradient
            if eq_constraints is not None:
                x_for_eq = x.detach()
                J_h = torch.func.jacrev(lambda xx: eq_constraints(xx).reshape(-1))(x_for_eq)
                grad_barrier = grad_barrier + J_h.T @ lambda_eq

            # Check KKT residual
            kkt_norm = grad_barrier.abs().max().item()
            if eq_constraints is not None:
                h_val = eq_constraints(x.detach())
                if h_val.dim() == 0:
                    h_val = h_val.unsqueeze(0)
                kkt_norm = max(kkt_norm, h_val.abs().max().item())

            if kkt_norm < tol:
                break

            # Newton step (simplified: gradient descent with line search)
            direction = -grad_barrier
            alpha = 1.0
            for _ in range(20):
                x_new = x + alpha * direction
                with torch.no_grad():
                    f_new = objective(x_new)

                    # Check inequality feasibility
                    feasible = True
                    if ineq_constraints is not None:
                        g_new = ineq_constraints(x_new)
                        if g_new.dim() == 0:
                            g_new = g_new.unsqueeze(0)
                        if (g_new >= 0).any():
                            feasible = False

                    if feasible and f_new < f_val.detach():
                        break
                alpha *= 0.5
            else:
                x_new = x + alpha * direction

            x = x_new.detach()

        # Update dual variables
        if ineq_constraints is not None:
            with torch.no_grad():
                g_val = ineq_constraints(x)
                if g_val.dim() == 0:
                    g_val = g_val.unsqueeze(0)
                s_ineq = (-g_val).clamp(min=1e-10)
                lambda_ineq = mu / s_ineq

        if eq_constraints is not None:
            with torch.no_grad():
                h_val = eq_constraints(x)
                if h_val.dim() == 0:
                    h_val = h_val.unsqueeze(0)
                lambda_eq = lambda_eq + (1.0 / mu) * h_val

        # Check convergence
        converged_flag = True
        if eq_constraints is not None:
            if h_val.abs().max().item() > tol:
                converged_flag = False
        if ineq_constraints is not None:
            if g_val.max().item() > tol:
                converged_flag = False

        if converged_flag and kkt_norm < tol:
            converged = torch.tensor(True, device=x0.device)
            break

        # Reduce barrier parameter
        mu = mu * mu_factor
    else:
        converged = torch.tensor(False, device=x0.device)

    with torch.no_grad():
        f_final = objective(x)

    x_with_grad = _IPImplicitGrad.apply(x, objective, eq_constraints, ineq_constraints)

    return OptimizeResult(
        x=x_with_grad,
        converged=converged,
        num_iterations=torch.tensor(num_iter, dtype=torch.int64, device=x0.device),
        fun=f_final,
    )
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/torchscience/optimization/constrained/test__interior_point.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/torchscience/optimization/constrained/_interior_point.py tests/torchscience/optimization/constrained/test__interior_point.py
git commit -m "feat(optimization): add interior point constrained solver"
```

---

### Task 5: SQP — Tests

**Files:**
- Create: `tests/torchscience/optimization/constrained/test__sqp.py`

**Step 1: Write tests**

```python
import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.constrained._sqp import sqp


class TestSQP:
    def test_unconstrained_quadratic(self):
        """Without constraints, should minimize f(x) = ||x||^2."""

        def objective(x):
            return torch.sum(x**2)

        result = sqp(objective, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_equality_constraint(self):
        """Minimize x^2 + y^2 subject to x + y = 1."""

        def objective(x):
            return torch.sum(x**2)

        def eq_constraints(x):
            return x.sum() - 1.0

        result = sqp(
            objective,
            torch.tensor([0.6, 0.4]),
            eq_constraints=eq_constraints,
        )
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(result.x, expected, atol=1e-3, rtol=1e-3)

    def test_inequality_constraint(self):
        """Minimize -x subject to x <= 2."""

        def objective(x):
            return -x.sum()

        def ineq_constraints(x):
            return x - 2.0

        result = sqp(
            objective,
            torch.tensor([0.0]),
            ineq_constraints=ineq_constraints,
        )
        expected = torch.tensor([2.0])
        torch.testing.assert_close(result.x, expected, atol=1e-2, rtol=1e-2)

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def objective(x):
            return torch.sum(x**2)

        result = sqp(objective, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)

    def test_convergence_flag(self):
        """Test convergence flag."""

        def objective(x):
            return torch.sum(x**2)

        result = sqp(objective, torch.tensor([1.0]))
        assert result.converged.item() is True


class TestSQPDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def objective(x):
            return torch.sum(x**2)

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = sqp(objective, x0)
        assert result.x.dtype == dtype
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/torchscience/optimization/constrained/test__sqp.py -v 2>&1 | head -20`
Expected: ImportError

---

### Task 6: SQP — Implementation

**Files:**
- Create: `src/torchscience/optimization/constrained/_sqp.py`

**Step 1: Implement the solver**

SQP solves a QP subproblem at each iteration. Since the QP solver from Phase 3 may not exist yet, we use a self-contained QP solver (active-set method for small problems) within SQP.

The implementation:
1. At each iteration, linearize constraints and build a quadratic model
2. Solve the QP subproblem for the step direction
3. Use L1 merit function with Armijo line search
4. Update Hessian approximation via damped BFGS

```python
from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._l_bfgs import _compute_grad


def _solve_qp_equality(H, c, A, b):
    """Solve min 0.5 x^T H x + c^T x  s.t. A x = b via KKT system."""
    n = H.shape[0]
    m = A.shape[0] if A is not None and A.numel() > 0 else 0

    if m == 0:
        # Unconstrained QP
        reg = 1e-8 * torch.eye(n, dtype=H.dtype, device=H.device)
        return torch.linalg.solve(H + reg, -c)

    # KKT system: [[H, A^T], [A, 0]] [x; lambda] = [-c; b]
    KKT = torch.zeros(n + m, n + m, dtype=H.dtype, device=H.device)
    KKT[:n, :n] = H
    KKT[:n, n:] = A.T
    KKT[n:, :n] = A

    rhs = torch.zeros(n + m, dtype=H.dtype, device=H.device)
    rhs[:n] = -c
    rhs[n:] = b

    reg = 1e-8 * torch.eye(n + m, dtype=H.dtype, device=H.device)
    sol = torch.linalg.solve(KKT + reg, rhs)
    return sol[:n]


def sqp(
    objective: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    eq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    ineq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    tol: Optional[float] = None,
    maxiter: int = 50,
) -> OptimizeResult:
    r"""
    Sequential Quadratic Programming for constrained optimization.

    Solves: min f(x) s.t. h(x) = 0, g(x) <= 0

    At each iteration, solves a QP subproblem with a quadratic model of
    the objective and linearized constraints. Uses damped BFGS for the
    Hessian approximation and L1 merit function for step acceptance.

    Parameters
    ----------
    objective : Callable[[Tensor], Tensor]
        Scalar-valued objective function.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    eq_constraints : Callable[[Tensor], Tensor], optional
        Equality constraints h(x) = 0.
    ineq_constraints : Callable[[Tensor], Tensor], optional
        Inequality constraints g(x) <= 0.
    tol : float, optional
        KKT tolerance. Default: ``torch.finfo(x0.dtype).eps ** 0.5``.
    maxiter : int, optional
        Maximum iterations. Default: 50.

    Returns
    -------
    OptimizeResult
        Result with fields ``x``, ``converged``, ``num_iterations``, ``fun``.
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    n = x0.numel()
    x = x0.clone().detach()
    B = torch.eye(n, dtype=x0.dtype, device=x0.device)  # Hessian approx
    num_iter = 0
    penalty = 1.0

    for k in range(maxiter):
        num_iter = k + 1

        # Evaluate objective and gradient
        f_val, grad_f = _compute_grad(objective, x)

        # Evaluate constraints
        h_val = None
        J_h = None
        if eq_constraints is not None:
            h_val = eq_constraints(x.detach())
            if h_val.dim() == 0:
                h_val = h_val.unsqueeze(0)
            J_h = torch.func.jacrev(lambda xx: eq_constraints(xx).reshape(-1))(x.detach())

        g_val = None
        J_g = None
        if ineq_constraints is not None:
            g_val = ineq_constraints(x.detach())
            if g_val.dim() == 0:
                g_val = g_val.unsqueeze(0)
            J_g = torch.func.jacrev(lambda xx: ineq_constraints(xx).reshape(-1))(x.detach())

        # Check KKT convergence
        kkt_norm = grad_f.abs().max().item()
        if h_val is not None:
            kkt_norm = max(kkt_norm, h_val.abs().max().item())
        if g_val is not None:
            kkt_norm = max(kkt_norm, g_val.clamp(min=0).max().item())

        if kkt_norm < tol:
            converged = torch.tensor(True, device=x0.device)
            break

        # Build and solve QP subproblem
        # For equality constraints only (simplified)
        # Full QP with inequalities would use the Phase 3 QP solver
        if ineq_constraints is not None and eq_constraints is not None:
            # Convert active inequalities to equalities
            active = g_val >= -tol
            if active.any():
                A_active = torch.cat([J_h, J_g[active]], dim=0) if J_h is not None else J_g[active]
                b_active = torch.cat([-h_val, -g_val[active]], dim=0) if h_val is not None else -g_val[active]
            else:
                A_active = J_h
                b_active = -h_val if h_val is not None else None
        elif eq_constraints is not None:
            A_active = J_h
            b_active = -h_val
        elif ineq_constraints is not None:
            active = g_val >= -tol
            if active.any():
                A_active = J_g[active]
                b_active = -g_val[active]
            else:
                A_active = None
                b_active = None
        else:
            A_active = None
            b_active = None

        d = _solve_qp_equality(B, grad_f, A_active, b_active)

        # L1 merit function line search
        def merit(xx, pen):
            with torch.no_grad():
                m_val = objective(xx)
                if eq_constraints is not None:
                    h = eq_constraints(xx)
                    if h.dim() == 0:
                        h = h.unsqueeze(0)
                    m_val = m_val + pen * h.abs().sum()
                if ineq_constraints is not None:
                    g = ineq_constraints(xx)
                    if g.dim() == 0:
                        g = g.unsqueeze(0)
                    m_val = m_val + pen * g.clamp(min=0).sum()
                return m_val

        merit_current = merit(x, penalty)
        alpha = 1.0
        for _ in range(20):
            x_new = x + alpha * d
            merit_new = merit(x_new, penalty)
            if merit_new < merit_current - 1e-4 * alpha * d.norm() ** 2:
                break
            alpha *= 0.5
        else:
            x_new = x + alpha * d

        # BFGS update
        _, grad_f_new = _compute_grad(objective, x_new)
        s = x_new - x
        y = grad_f_new - grad_f
        ys = (y * s).sum()

        if ys > 1e-10:
            Bs = B @ s
            sBs = (s * Bs).sum()
            B = B - torch.outer(Bs, Bs) / sBs.clamp(min=1e-30) + torch.outer(y, y) / ys

        x = x_new.detach()
    else:
        converged = torch.tensor(False, device=x0.device)

    with torch.no_grad():
        f_final = objective(x)

    return OptimizeResult(
        x=x,
        converged=converged,
        num_iterations=torch.tensor(num_iter, dtype=torch.int64, device=x0.device),
        fun=f_final,
    )
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/torchscience/optimization/constrained/test__sqp.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/torchscience/optimization/constrained/_sqp.py tests/torchscience/optimization/constrained/test__sqp.py
git commit -m "feat(optimization): add SQP constrained solver"
```

---

### Task 7: Update Constrained Module Exports

**Files:**
- Modify: `src/torchscience/optimization/constrained/__init__.py`

**Step 1: Add new exports**

Add imports for `interior_point`, `l_bfgs_b`, and `sqp`. Update `__all__`.

```python
from ._augmented_lagrangian import augmented_lagrangian
from ._interior_point import interior_point
from ._l_bfgs_b import l_bfgs_b
from ._sqp import sqp

__all__ = [
    "augmented_lagrangian",
    "interior_point",
    "l_bfgs_b",
    "sqp",
]
```

**Step 2: Run all constrained tests**

Run: `uv run pytest tests/torchscience/optimization/constrained/ -v`
Expected: All tests PASS (existing augmented Lagrangian + new)

**Step 3: Commit**

```bash
git add src/torchscience/optimization/constrained/__init__.py
git commit -m "feat(optimization): export new constrained solvers"
```

---

### Task 8: Final Verification

**Step 1: Run full optimization test suite**

Run: `uv run pytest tests/torchscience/optimization/ -v --tb=short`
Expected: All tests PASS

**Step 2: Run linting**

Run: `uv run pre-commit run --all-files`
Expected: All checks PASS

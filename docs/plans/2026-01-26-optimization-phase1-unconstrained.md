# Optimization Phase 1: Unconstrained Minimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add four new unconstrained minimization solvers (Newton-CG, trust-region, conjugate gradient, Nelder-Mead) and expand the `minimize()` dispatch interface.

**Architecture:** Each solver is a standalone Python function in `src/torchscience/optimization/minimization/` returning `OptimizeResult`. They reuse existing line search utilities and the implicit differentiation pattern from L-BFGS. The `minimize()` function dispatches to all solvers via a `method` string parameter.

**Tech Stack:** PyTorch (torch.func for Hessian-vector products, torch.autograd for implicit diff, torch.linalg.solve for linear systems)

---

### Task 1: Nonlinear Conjugate Gradient — Tests

**Files:**
- Create: `tests/torchscience/optimization/minimization/test__conjugate_gradient.py`

**Step 1: Write tests**

```python
import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._conjugate_gradient import conjugate_gradient


class TestConjugateGradient:
    def test_quadratic(self):
        """Minimize f(x) = ||x||^2."""

        def f(x):
            return (x**2).sum()

        result = conjugate_gradient(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_rosenbrock_2d(self):
        """Minimize 2D Rosenbrock function."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = conjugate_gradient(rosenbrock, torch.tensor([-1.0, 1.0]), maxiter=500)
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 1.0]),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_convergence_flag(self):
        """Test that convergence flag is set correctly."""

        def f(x):
            return (x**2).sum()

        result = conjugate_gradient(f, torch.tensor([1.0]))
        assert result.converged.item() is True

    def test_convergence_flag_not_converged(self):
        """Test convergence flag when maxiter is too low."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = conjugate_gradient(rosenbrock, torch.tensor([-5.0, 5.0]), maxiter=2)
        assert isinstance(result.converged, torch.Tensor)

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def f(x):
            return (x**2).sum()

        result = conjugate_gradient(f, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.num_iterations is not None
        assert result.fun is not None

    def test_fun_value(self):
        """Test that fun contains the objective value at the solution."""

        def f(x):
            return (x**2).sum()

        result = conjugate_gradient(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.fun,
            torch.tensor(0.0),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_explicit_gradient(self):
        """Test with explicit gradient function."""

        def f(x):
            return (x**2).sum()

        def grad_f(x):
            return 2 * x

        result = conjugate_gradient(f, torch.tensor([3.0, 4.0]), grad=grad_f)
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_fletcher_reeves_variant(self):
        """Test Fletcher-Reeves variant."""

        def f(x):
            return (x**2).sum()

        result = conjugate_gradient(
            f, torch.tensor([3.0, 4.0]), variant="fletcher-reeves"
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_hestenes_stiefel_variant(self):
        """Test Hestenes-Stiefel variant."""

        def f(x):
            return (x**2).sum()

        result = conjugate_gradient(
            f, torch.tensor([3.0, 4.0]), variant="hestenes-stiefel"
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_tol_parameter(self):
        """Test that tol parameter affects convergence."""

        def f(x):
            return (x**2).sum()

        result = conjugate_gradient(f, torch.tensor([1.0]), tol=1e-12)
        assert result.x.abs().item() < 1e-6


class TestConjugateGradientAutograd:
    def test_implicit_diff_quadratic(self):
        """Test implicit differentiation through a quadratic."""
        target = torch.tensor([5.0, 3.0], requires_grad=True)

        def f(x):
            return ((x - target) ** 2).sum()

        result = conjugate_gradient(f, torch.zeros(2))
        result.x.sum().backward()

        torch.testing.assert_close(
            target.grad,
            torch.ones(2),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_numerical_gradient(self):
        """Test implicit gradient against numerical finite differences."""
        eps = 1e-4
        theta = torch.tensor([2.0], requires_grad=True)

        def f(x):
            return ((x - theta) ** 2).sum()

        result = conjugate_gradient(f, torch.tensor([0.0]))
        result.x.sum().backward()
        analytic_grad = theta.grad.clone()

        with torch.no_grad():
            theta_plus = theta + eps
            theta_minus = theta - eps

        def f_plus(x):
            return ((x - theta_plus) ** 2).sum()

        def f_minus(x):
            return ((x - theta_minus) ** 2).sum()

        r_plus = conjugate_gradient(f_plus, torch.tensor([0.0]))
        r_minus = conjugate_gradient(f_minus, torch.tensor([0.0]))
        numerical_grad = (r_plus.x.sum() - r_minus.x.sum()) / (2 * eps)

        torch.testing.assert_close(
            analytic_grad,
            numerical_grad.unsqueeze(0),
            atol=1e-2,
            rtol=1e-2,
        )


class TestConjugateGradientDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def f(x):
            return (x**2).sum()

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = conjugate_gradient(f, x0)
        assert result.x.dtype == dtype
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/torchscience/optimization/minimization/test__conjugate_gradient.py -v 2>&1 | head -20`
Expected: ImportError — `_conjugate_gradient` module does not exist

---

### Task 2: Nonlinear Conjugate Gradient — Implementation

**Files:**
- Create: `src/torchscience/optimization/minimization/_conjugate_gradient.py`

**Step 1: Implement the solver**

```python
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
                _f_and_grad, x, d, f_val, g,
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
        converged = torch.tensor(g.abs().max() < tol, device=x.device)

    with torch.no_grad():
        f_final = fun(x)

    x_with_grad = _implicit_diff_step(fun, x)

    return OptimizeResult(
        x=x_with_grad,
        converged=converged,
        num_iterations=torch.tensor(num_iter, dtype=torch.int64, device=x.device),
        fun=f_final,
    )
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/torchscience/optimization/minimization/test__conjugate_gradient.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/torchscience/optimization/minimization/_conjugate_gradient.py tests/torchscience/optimization/minimization/test__conjugate_gradient.py
git commit -m "feat(optimization): add nonlinear conjugate gradient solver"
```

---

### Task 3: Newton-CG — Tests

**Files:**
- Create: `tests/torchscience/optimization/minimization/test__newton_cg.py`

**Step 1: Write tests**

```python
import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._newton_cg import newton_cg


class TestNewtonCG:
    def test_quadratic(self):
        """Minimize f(x) = ||x||^2."""

        def f(x):
            return (x**2).sum()

        result = newton_cg(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_rosenbrock_2d(self):
        """Minimize 2D Rosenbrock function."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = newton_cg(rosenbrock, torch.tensor([-1.0, 1.0]), maxiter=200)
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 1.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_convergence_flag(self):
        """Test that convergence flag is set correctly."""

        def f(x):
            return (x**2).sum()

        result = newton_cg(f, torch.tensor([1.0]))
        assert result.converged.item() is True

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def f(x):
            return (x**2).sum()

        result = newton_cg(f, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.num_iterations is not None
        assert result.fun is not None

    def test_fun_value(self):
        """Test that fun contains the objective value at the solution."""

        def f(x):
            return (x**2).sum()

        result = newton_cg(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.fun,
            torch.tensor(0.0),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_ill_conditioned_quadratic(self):
        """Newton-CG handles ill-conditioned problems better than CG."""

        A = torch.diag(torch.tensor([1.0, 1000.0]))

        def f(x):
            return 0.5 * x @ A @ x

        result = newton_cg(f, torch.tensor([10.0, 10.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_explicit_gradient(self):
        """Test with explicit gradient function."""

        def f(x):
            return (x**2).sum()

        def grad_f(x):
            return 2 * x

        result = newton_cg(f, torch.tensor([3.0, 4.0]), grad=grad_f)
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_tol_parameter(self):
        """Test that tol parameter affects convergence."""

        def f(x):
            return (x**2).sum()

        result = newton_cg(f, torch.tensor([1.0]), tol=1e-12)
        assert result.x.abs().item() < 1e-6


class TestNewtonCGAutograd:
    def test_implicit_diff_quadratic(self):
        """Test implicit differentiation through a quadratic."""
        target = torch.tensor([5.0, 3.0], requires_grad=True)

        def f(x):
            return ((x - target) ** 2).sum()

        result = newton_cg(f, torch.zeros(2))
        result.x.sum().backward()

        torch.testing.assert_close(
            target.grad,
            torch.ones(2),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_numerical_gradient(self):
        """Test implicit gradient against numerical finite differences."""
        eps = 1e-4
        theta = torch.tensor([2.0], requires_grad=True)

        def f(x):
            return ((x - theta) ** 2).sum()

        result = newton_cg(f, torch.tensor([0.0]))
        result.x.sum().backward()
        analytic_grad = theta.grad.clone()

        with torch.no_grad():
            theta_plus = theta + eps
            theta_minus = theta - eps

        def f_plus(x):
            return ((x - theta_plus) ** 2).sum()

        def f_minus(x):
            return ((x - theta_minus) ** 2).sum()

        r_plus = newton_cg(f_plus, torch.tensor([0.0]))
        r_minus = newton_cg(f_minus, torch.tensor([0.0]))
        numerical_grad = (r_plus.x.sum() - r_minus.x.sum()) / (2 * eps)

        torch.testing.assert_close(
            analytic_grad,
            numerical_grad.unsqueeze(0),
            atol=1e-2,
            rtol=1e-2,
        )


class TestNewtonCGDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def f(x):
            return (x**2).sum()

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = newton_cg(f, x0)
        assert result.x.dtype == dtype
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/torchscience/optimization/minimization/test__newton_cg.py -v 2>&1 | head -20`
Expected: ImportError

---

### Task 4: Newton-CG — Implementation

**Files:**
- Create: `src/torchscience/optimization/minimization/_newton_cg.py`

**Step 1: Implement the solver**

```python
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
    """Compute H @ v using forward-over-reverse AD (torch.func.jvp)."""
    def grad_fn(x_inner):
        x_inner = x_inner.detach().requires_grad_(True)
        f_val = fun(x_inner)
        if f_val.dim() > 0:
            f_val = f_val.sum()
        g = torch.autograd.grad(f_val, x_inner, create_graph=True)[0]
        return g

    _, hvp = torch.func.jvp(grad_fn, (x,), (v,))
    return hvp.detach()


def _cg_steihaug(hvp_fn, grad, tol_cg, max_cg_iter):
    """Solve H @ d = -grad approximately using CG with negative curvature check."""
    n = grad.numel()
    d = torch.zeros_like(grad)
    r = grad.clone()
    p = -r.clone()

    r_dot_r = (r * r).sum()
    tol_sq = tol_cg ** 2

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
                _f_and_grad, x, direction, f_val, g,
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
        num_iterations=torch.tensor(num_iter, dtype=torch.int64, device=x.device),
        fun=f_final,
    )
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/torchscience/optimization/minimization/test__newton_cg.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/torchscience/optimization/minimization/_newton_cg.py tests/torchscience/optimization/minimization/test__newton_cg.py
git commit -m "feat(optimization): add Newton-CG (truncated Newton) solver"
```

---

### Task 5: Trust-Region Newton — Tests

**Files:**
- Create: `tests/torchscience/optimization/minimization/test__trust_region.py`

**Step 1: Write tests**

```python
import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._trust_region import trust_region


class TestTrustRegion:
    def test_quadratic(self):
        """Minimize f(x) = ||x||^2."""

        def f(x):
            return (x**2).sum()

        result = trust_region(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_rosenbrock_2d(self):
        """Minimize 2D Rosenbrock function."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = trust_region(rosenbrock, torch.tensor([-1.0, 1.0]), maxiter=200)
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 1.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_convergence_flag(self):
        """Test that convergence flag is set correctly."""

        def f(x):
            return (x**2).sum()

        result = trust_region(f, torch.tensor([1.0]))
        assert result.converged.item() is True

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def f(x):
            return (x**2).sum()

        result = trust_region(f, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.num_iterations is not None
        assert result.fun is not None

    def test_fun_value(self):
        """Test that fun contains the objective value at the solution."""

        def f(x):
            return (x**2).sum()

        result = trust_region(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.fun,
            torch.tensor(0.0),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_ill_conditioned_quadratic(self):
        """Trust-region handles ill-conditioned problems."""

        A = torch.diag(torch.tensor([1.0, 1000.0]))

        def f(x):
            return 0.5 * x @ A @ x

        result = trust_region(f, torch.tensor([10.0, 10.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_tol_parameter(self):
        """Test that tol parameter affects convergence."""

        def f(x):
            return (x**2).sum()

        result = trust_region(f, torch.tensor([1.0]), tol=1e-12)
        assert result.x.abs().item() < 1e-6

    def test_initial_trust_radius(self):
        """Test custom initial trust radius."""

        def f(x):
            return (x**2).sum()

        result = trust_region(
            f, torch.tensor([3.0, 4.0]), initial_trust_radius=0.1
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )


class TestTrustRegionAutograd:
    def test_implicit_diff_quadratic(self):
        """Test implicit differentiation through a quadratic."""
        target = torch.tensor([5.0, 3.0], requires_grad=True)

        def f(x):
            return ((x - target) ** 2).sum()

        result = trust_region(f, torch.zeros(2))
        result.x.sum().backward()

        torch.testing.assert_close(
            target.grad,
            torch.ones(2),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_numerical_gradient(self):
        """Test implicit gradient against numerical finite differences."""
        eps = 1e-4
        theta = torch.tensor([2.0], requires_grad=True)

        def f(x):
            return ((x - theta) ** 2).sum()

        result = trust_region(f, torch.tensor([0.0]))
        result.x.sum().backward()
        analytic_grad = theta.grad.clone()

        with torch.no_grad():
            theta_plus = theta + eps
            theta_minus = theta - eps

        def f_plus(x):
            return ((x - theta_plus) ** 2).sum()

        def f_minus(x):
            return ((x - theta_minus) ** 2).sum()

        r_plus = trust_region(f_plus, torch.tensor([0.0]))
        r_minus = trust_region(f_minus, torch.tensor([0.0]))
        numerical_grad = (r_plus.x.sum() - r_minus.x.sum()) / (2 * eps)

        torch.testing.assert_close(
            analytic_grad,
            numerical_grad.unsqueeze(0),
            atol=1e-2,
            rtol=1e-2,
        )


class TestTrustRegionDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def f(x):
            return (x**2).sum()

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = trust_region(f, x0)
        assert result.x.dtype == dtype
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/torchscience/optimization/minimization/test__trust_region.py -v 2>&1 | head -20`
Expected: ImportError

---

### Task 6: Trust-Region Newton — Implementation

**Files:**
- Create: `src/torchscience/optimization/minimization/_trust_region.py`

**Step 1: Implement the solver**

```python
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
            discriminant = dp ** 2 - pp * (dd - trust_radius ** 2)
            tau = (-dp + torch.sqrt(discriminant.clamp(min=0))) / pp.clamp(min=1e-30)
            return d + tau * p

        alpha = r_dot_r / pHp
        d_new = d + alpha * p

        # Check trust-region boundary
        if d_new.norm() >= trust_radius:
            # Step to boundary
            dd = (d * d).sum()
            dp = (d * p).sum()
            pp = (p * p).sum()
            discriminant = dp ** 2 - pp * (dd - trust_radius ** 2)
            tau = (-dp + torch.sqrt(discriminant.clamp(min=0))) / pp.clamp(min=1e-30)
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
        rho = actual / predicted.clamp(min=1e-30) if predicted > 0 else torch.tensor(0.0)
        if isinstance(rho, Tensor):
            rho = rho.item()

        # Update trust radius
        step_norm = step.norm().item()
        if rho < 0.25:
            trust_radius = 0.25 * step_norm
        elif rho > 0.75 and abs(step_norm - trust_radius) < 1e-10:
            trust_radius = min(2.0 * trust_radius, max_trust_radius)

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
        num_iterations=torch.tensor(num_iter, dtype=torch.int64, device=x.device),
        fun=f_final,
    )
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/torchscience/optimization/minimization/test__trust_region.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/torchscience/optimization/minimization/_trust_region.py tests/torchscience/optimization/minimization/test__trust_region.py
git commit -m "feat(optimization): add trust-region Newton solver"
```

---

### Task 7: Nelder-Mead — Tests

**Files:**
- Create: `tests/torchscience/optimization/minimization/test__nelder_mead.py`

**Step 1: Write tests**

```python
import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._nelder_mead import nelder_mead


class TestNelderMead:
    def test_quadratic(self):
        """Minimize f(x) = ||x||^2."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_rosenbrock_2d(self):
        """Minimize 2D Rosenbrock function."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = nelder_mead(rosenbrock, torch.tensor([-1.0, 1.0]), maxiter=1000)
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 1.0]),
            atol=0.1,
            rtol=0.1,
        )

    def test_convergence_flag(self):
        """Test that convergence flag is set correctly."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([1.0, 1.0]))
        assert result.converged.item() is True

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.num_iterations is not None
        assert result.fun is not None

    def test_fun_value(self):
        """Test that fun contains the objective value at the solution."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([3.0, 4.0]))
        assert result.fun.item() < 0.01

    def test_no_gradient_required(self):
        """Nelder-Mead works on non-differentiable functions."""

        def f(x):
            return x.abs().sum()

        result = nelder_mead(f, torch.tensor([3.0, -4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_result_is_detached(self):
        """Nelder-Mead result has no gradient (derivative-free method)."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([1.0]))
        assert not result.x.requires_grad

    def test_tol_parameter(self):
        """Test that tol affects convergence."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([1.0, 1.0]), tol=1e-10)
        assert result.fun.item() < 1e-8

    def test_1d(self):
        """Test 1D optimization."""

        def f(x):
            return (x - 3) ** 2

        result = nelder_mead(f, torch.tensor([0.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([3.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_booth_function(self):
        """Test Booth function: (x + 2y - 7)^2 + (2x + y - 5)^2."""

        def booth(x):
            return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        result = nelder_mead(booth, torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 3.0]),
            atol=1e-3,
            rtol=1e-3,
        )


class TestNelderMeadDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def f(x):
            return (x**2).sum()

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = nelder_mead(f, x0)
        assert result.x.dtype == dtype
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/torchscience/optimization/minimization/test__nelder_mead.py -v 2>&1 | head -20`
Expected: ImportError

---

### Task 8: Nelder-Mead — Implementation

**Files:**
- Create: `src/torchscience/optimization/minimization/_nelder_mead.py`

**Step 1: Implement the solver**

```python
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

    # Evaluate function at all vertices
    with torch.no_grad():
        f_values = torch.stack([fun(simplex[i]) for i in range(n + 1)])

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
            f_r = fun(x_r)

        if f_values[0] <= f_r < f_values[-2]:
            # Accept reflection
            simplex[-1] = x_r
            f_values[-1] = f_r
            continue

        if f_r < f_values[0]:
            # Try expansion
            x_e = centroid + gamma * (x_r - centroid)
            with torch.no_grad():
                f_e = fun(x_e)
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
                f_c = fun(x_c)
            if f_c <= f_r:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue
        else:
            # Inside contraction
            x_c = centroid + rho * (x_worst - centroid)
            with torch.no_grad():
                f_c = fun(x_c)
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue

        # Shrink
        best = simplex[0].clone()
        for i in range(1, n + 1):
            simplex[i] = best + sigma * (simplex[i] - best)
            with torch.no_grad():
                f_values[i] = fun(simplex[i])
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
        num_iterations=torch.tensor(num_iter, dtype=torch.int64, device=x0.device),
        fun=f_best,
    )
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/torchscience/optimization/minimization/test__nelder_mead.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/torchscience/optimization/minimization/_nelder_mead.py tests/torchscience/optimization/minimization/test__nelder_mead.py
git commit -m "feat(optimization): add Nelder-Mead derivative-free solver"
```

---

### Task 9: Update minimize() Dispatch and Module Exports

**Files:**
- Modify: `src/torchscience/optimization/minimization/_minimize.py`
- Modify: `src/torchscience/optimization/minimization/__init__.py`

**Step 1: Update `_minimize.py` to dispatch to new solvers**

Add imports and dispatch branches after existing methods. The `minimize()` function should handle these new method strings:
- `"conjugate-gradient"` → `conjugate_gradient()`
- `"newton-cg"` → `newton_cg()`
- `"trust-region"` → `trust_region()`
- `"nelder-mead"` → `nelder_mead()`

For each new method, filter `**kwargs` to only pass recognized parameters (same pattern as existing LM handling). Update the error message to list all supported methods.

**Step 2: Update `__init__.py` to export new solvers**

Add imports:
```python
from ._conjugate_gradient import conjugate_gradient
from ._newton_cg import newton_cg
from ._nelder_mead import nelder_mead
from ._trust_region import trust_region
```

Add to `__all__`:
```python
__all__ = [
    "conjugate_gradient",
    "curve_fit",
    "l_bfgs",
    "levenberg_marquardt",
    "minimize",
    "nelder_mead",
    "newton_cg",
    "trust_region",
]
```

**Step 3: Write a test for the expanded minimize() dispatch**

Create `tests/torchscience/optimization/minimization/test__minimize_dispatch.py`:

```python
import pytest
import torch
import torch.testing

from torchscience.optimization.minimization import minimize


class TestMinimizeDispatch:
    @pytest.mark.parametrize(
        "method",
        [
            "l-bfgs",
            "conjugate-gradient",
            "newton-cg",
            "trust-region",
            "nelder-mead",
        ],
    )
    def test_quadratic_all_methods(self, method):
        """All methods should solve a simple quadratic."""

        def f(x):
            return (x**2).sum()

        result = minimize(f, torch.tensor([3.0, 4.0]), method=method)
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""

        def f(x):
            return (x**2).sum()

        with pytest.raises(ValueError, match="Unknown method"):
            minimize(f, torch.tensor([1.0]), method="unknown")
```

**Step 4: Run all optimization tests**

Run: `uv run pytest tests/torchscience/optimization/ -v`
Expected: All tests PASS (existing + new)

**Step 5: Commit**

```bash
git add src/torchscience/optimization/minimization/_minimize.py src/torchscience/optimization/minimization/__init__.py tests/torchscience/optimization/minimization/test__minimize_dispatch.py
git commit -m "feat(optimization): expand minimize() with CG, Newton-CG, trust-region, Nelder-Mead"
```

---

### Task 10: Final Verification

**Step 1: Run the full optimization test suite**

Run: `uv run pytest tests/torchscience/optimization/ -v --tb=short`
Expected: All tests PASS (existing 159 + new tests)

**Step 2: Run linting**

Run: `uv run pre-commit run --all-files`
Expected: All checks PASS

**Step 3: Verify git log**

Run: `git log --oneline -5`
Expected: Four clean commits for the four new solvers + dispatch update

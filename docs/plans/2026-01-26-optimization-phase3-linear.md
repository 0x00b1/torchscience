# Optimization Phase 3: Linear Algebra-Based Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a new `optimization/linear/` submodule with linear programming, quadratic programming, and nonlinear least squares solvers.

**Architecture:** New submodule at `src/torchscience/optimization/linear/`. LP and QP use primal-dual interior point methods. Least squares wraps existing Levenberg-Marquardt with an `OptimizeResult` interface and adds bounded least squares. All support implicit differentiation through KKT conditions.

**Tech Stack:** PyTorch (torch.linalg.solve for KKT systems, torch.func.jacrev for Jacobians, torch.autograd.Function for implicit diff)

---

### Task 1: Quadratic Programming — Tests

QP is implemented first because SQP (Phase 2) depends on it.

**Files:**
- Create: `tests/torchscience/optimization/linear/test__quadratic_program.py`

**Step 1: Write tests**

```python
import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.linear._quadratic_program import quadratic_program


class TestQuadraticProgram:
    def test_unconstrained(self):
        """min 0.5 x^T Q x + c^T x with Q = I, c = [-1, -2]."""
        Q = torch.eye(2)
        c = torch.tensor([-1.0, -2.0])
        result = quadratic_program(Q, c)
        # Solution: x = Q^{-1} (-c) = [1, 2]
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 2.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_equality_constraint(self):
        """min 0.5 x^T I x subject to x1 + x2 = 1."""
        Q = torch.eye(2)
        c = torch.zeros(2)
        A_eq = torch.tensor([[1.0, 1.0]])
        b_eq = torch.tensor([1.0])
        result = quadratic_program(Q, c, A_eq=A_eq, b_eq=b_eq)
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(result.x, expected, atol=1e-3, rtol=1e-3)

    def test_inequality_constraint(self):
        """min 0.5 x^T I x - [2, 2]^T x subject to x1 + x2 <= 1."""
        Q = torch.eye(2)
        c = torch.tensor([-2.0, -2.0])
        A_ub = torch.tensor([[1.0, 1.0]])
        b_ub = torch.tensor([1.0])
        result = quadratic_program(Q, c, A_ub=A_ub, b_ub=b_ub)
        # Unconstrained optimum is [2, 2], but constraint pushes to [0.5, 0.5]
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(result.x, expected, atol=1e-2, rtol=1e-2)

    def test_result_type(self):
        """Test that result is an OptimizeResult."""
        Q = torch.eye(2)
        c = torch.zeros(2)
        result = quadratic_program(Q, c)
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.fun is not None

    def test_fun_value(self):
        """Test objective value at solution."""
        Q = torch.eye(2)
        c = torch.tensor([-1.0, -2.0])
        result = quadratic_program(Q, c)
        # f(x*) = 0.5 * [1,2] @ I @ [1,2] + [-1,-2] @ [1,2] = 2.5 - 5 = -2.5
        torch.testing.assert_close(
            result.fun,
            torch.tensor(-2.5),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_positive_definite(self):
        """Non-identity PSD matrix."""
        Q = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        c = torch.tensor([-1.0, -1.0])
        result = quadratic_program(Q, c)
        # x* = Q^{-1} (-c)
        expected = torch.linalg.solve(Q, -c)
        torch.testing.assert_close(result.x, expected, atol=1e-4, rtol=1e-4)


class TestQuadraticProgramAutograd:
    def test_implicit_diff_cost(self):
        """Gradient w.r.t. cost vector c."""
        Q = torch.eye(2)
        c = torch.tensor([-1.0, -2.0], requires_grad=True)
        result = quadratic_program(Q, c)
        result.x.sum().backward()
        assert c.grad is not None
        # dx*/dc = -Q^{-1} = -I, so d(sum(x*))/dc = [-1, -1]
        torch.testing.assert_close(
            c.grad,
            torch.tensor([-1.0, -1.0]),
            atol=1e-2,
            rtol=1e-2,
        )


class TestQuadraticProgramDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""
        Q = torch.eye(2, dtype=dtype)
        c = torch.zeros(2, dtype=dtype)
        result = quadratic_program(Q, c)
        assert result.x.dtype == dtype
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/torchscience/optimization/linear/test__quadratic_program.py -v 2>&1 | head -20`
Expected: ImportError

---

### Task 2: Quadratic Programming — Implementation

**Files:**
- Create: `src/torchscience/optimization/linear/__init__.py`
- Create: `src/torchscience/optimization/linear/_quadratic_program.py`

**Step 1: Create `__init__.py`**

```python
from ._quadratic_program import quadratic_program

__all__ = [
    "quadratic_program",
]
```

**Step 2: Implement QP solver**

Uses primal-dual interior point method for QP with inequality constraints, and direct KKT solve for equality-only problems.

```python
from typing import Optional

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult


class _QPImplicitGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_opt, Q, c, A_ub, b_ub, A_eq, b_eq):
        ctx.save_for_backward(x_opt, Q, c, A_ub, b_ub, A_eq, b_eq)
        return x_opt.clone()

    @staticmethod
    def backward(ctx, grad_output):
        x_opt, Q, c, A_ub, b_ub, A_eq, b_eq = ctx.saved_tensors
        n = x_opt.shape[0]

        # Solve KKT system for implicit gradient
        # For unconstrained: dx*/dc = -Q^{-1}, dx*/dQ = Q^{-1} x* x*^T Q^{-1}
        reg = 1e-8 * torch.eye(n, dtype=Q.dtype, device=Q.device)
        v = torch.linalg.solve(Q + reg, grad_output)

        grad_Q = -0.5 * (torch.outer(v, x_opt) + torch.outer(x_opt, v))
        grad_c = -v

        return None, grad_Q, grad_c, None, None, None, None


def quadratic_program(
    Q: Tensor,
    c: Tensor,
    A_ub: Optional[Tensor] = None,
    b_ub: Optional[Tensor] = None,
    A_eq: Optional[Tensor] = None,
    b_eq: Optional[Tensor] = None,
    *,
    maxiter: int = 1000,
    tolerance: float = 1e-8,
) -> OptimizeResult:
    r"""
    Quadratic programming via primal-dual interior point method.

    Solves: min 0.5 x^T Q x + c^T x
    subject to: A_ub @ x <= b_ub, A_eq @ x = b_eq

    Parameters
    ----------
    Q : Tensor
        Positive semi-definite Hessian of shape ``(n, n)``.
    c : Tensor
        Linear cost vector of shape ``(n,)``.
    A_ub : Tensor, optional
        Inequality constraint matrix of shape ``(m_ub, n)``.
    b_ub : Tensor, optional
        Inequality constraint RHS of shape ``(m_ub,)``.
    A_eq : Tensor, optional
        Equality constraint matrix of shape ``(m_eq, n)``.
    b_eq : Tensor, optional
        Equality constraint RHS of shape ``(m_eq,)``.
    maxiter : int, optional
        Maximum iterations. Default: 1000.
    tolerance : float, optional
        Convergence tolerance. Default: 1e-8.

    Returns
    -------
    OptimizeResult
        Result with ``x`` (optimal point), ``converged``, ``num_iterations``,
        ``fun`` (optimal value).
    """
    n = c.shape[0]
    dtype = c.dtype
    device = c.device
    has_ub = A_ub is not None and b_ub is not None
    has_eq = A_eq is not None and b_eq is not None

    Q_det = Q.detach()
    c_det = c.detach()

    if not has_ub and not has_eq:
        # Unconstrained QP: x* = -Q^{-1} c
        reg = 1e-8 * torch.eye(n, dtype=dtype, device=device)
        x = torch.linalg.solve(Q_det + reg, -c_det)
        f_val = 0.5 * x @ Q_det @ x + c_det @ x

        # Dummy tensors for autograd function
        A_ub_dummy = torch.empty(0, n, dtype=dtype, device=device)
        b_ub_dummy = torch.empty(0, dtype=dtype, device=device)
        A_eq_dummy = torch.empty(0, n, dtype=dtype, device=device)
        b_eq_dummy = torch.empty(0, dtype=dtype, device=device)

        x_with_grad = _QPImplicitGrad.apply(
            x, Q, c, A_ub_dummy, b_ub_dummy, A_eq_dummy, b_eq_dummy
        )

        return OptimizeResult(
            x=x_with_grad,
            converged=torch.tensor(True, device=device),
            num_iterations=torch.tensor(1, dtype=torch.int64, device=device),
            fun=f_val,
        )

    if has_eq and not has_ub:
        # Equality-constrained QP: solve KKT system
        m_eq = A_eq.shape[0]
        A_eq_det = A_eq.detach()
        b_eq_det = b_eq.detach()

        KKT = torch.zeros(n + m_eq, n + m_eq, dtype=dtype, device=device)
        KKT[:n, :n] = Q_det
        KKT[:n, n:] = A_eq_det.T
        KKT[n:, :n] = A_eq_det

        rhs = torch.zeros(n + m_eq, dtype=dtype, device=device)
        rhs[:n] = -c_det
        rhs[n:] = b_eq_det

        reg = 1e-8 * torch.eye(n + m_eq, dtype=dtype, device=device)
        sol = torch.linalg.solve(KKT + reg, rhs)
        x = sol[:n]
        f_val = 0.5 * x @ Q_det @ x + c_det @ x

        A_ub_dummy = torch.empty(0, n, dtype=dtype, device=device)
        b_ub_dummy = torch.empty(0, dtype=dtype, device=device)

        x_with_grad = _QPImplicitGrad.apply(
            x, Q, c, A_ub_dummy, b_ub_dummy, A_eq, b_eq
        )

        return OptimizeResult(
            x=x_with_grad,
            converged=torch.tensor(True, device=device),
            num_iterations=torch.tensor(1, dtype=torch.int64, device=device),
            fun=f_val,
        )

    # General case: primal-dual interior point
    A_ub_det = A_ub.detach() if has_ub else torch.empty(0, n, dtype=dtype, device=device)
    b_ub_det = b_ub.detach() if has_ub else torch.empty(0, dtype=dtype, device=device)
    A_eq_det = A_eq.detach() if has_eq else torch.empty(0, n, dtype=dtype, device=device)
    b_eq_det = b_eq.detach() if has_eq else torch.empty(0, dtype=dtype, device=device)
    m_ub = A_ub_det.shape[0]
    m_eq = A_eq_det.shape[0]

    # Initialize: feasible point
    x = torch.zeros(n, dtype=dtype, device=device)
    s = torch.ones(m_ub, dtype=dtype, device=device)  # slacks
    lam_ub = torch.ones(m_ub, dtype=dtype, device=device)  # dual for ub
    lam_eq = torch.zeros(m_eq, dtype=dtype, device=device) if m_eq > 0 else torch.empty(0, dtype=dtype, device=device)

    # Make initial point feasible for inequalities
    if m_ub > 0:
        residual = A_ub_det @ x - b_ub_det
        s = (-residual).clamp(min=0.01)

    num_iter = 0

    for k in range(maxiter):
        num_iter = k + 1

        # Residuals
        # r_dual = Qx + c + A_ub^T lam_ub + A_eq^T lam_eq
        r_dual = Q_det @ x + c_det
        if m_ub > 0:
            r_dual = r_dual + A_ub_det.T @ lam_ub
        if m_eq > 0:
            r_dual = r_dual + A_eq_det.T @ lam_eq

        r_primal_ub = (A_ub_det @ x + s - b_ub_det) if m_ub > 0 else torch.empty(0, dtype=dtype, device=device)
        r_primal_eq = (A_eq_det @ x - b_eq_det) if m_eq > 0 else torch.empty(0, dtype=dtype, device=device)

        mu = (s * lam_ub).sum() / m_ub if m_ub > 0 else torch.tensor(0.0)

        # Check convergence
        res_norm = r_dual.abs().max().item()
        if m_ub > 0:
            res_norm = max(res_norm, r_primal_ub.abs().max().item())
        if m_eq > 0:
            res_norm = max(res_norm, r_primal_eq.abs().max().item())
        if m_ub > 0:
            res_norm = max(res_norm, mu.item())

        if res_norm < tolerance:
            break

        # Centering parameter
        sigma = 0.3
        mu_target = sigma * mu if m_ub > 0 else torch.tensor(0.0)

        # Complementarity residual
        r_comp = s * lam_ub - mu_target if m_ub > 0 else torch.empty(0, dtype=dtype, device=device)

        # Solve Newton system (reduced form)
        # Eliminate s and lam_ub:
        # ds = -s + S^{-1}(mu_target - lam_ub * ds_from_x)
        if m_ub > 0:
            S_inv = 1.0 / s.clamp(min=1e-12)
            D = lam_ub * S_inv  # diagonal scaling
            # Reduced system: (Q + A_ub^T D A_ub) dx = -(r_dual - A_ub^T S_inv r_comp + A_ub^T D r_primal_ub)
            M = Q_det + A_ub_det.T @ torch.diag(D) @ A_ub_det
            rhs_x = -(r_dual - A_ub_det.T @ (S_inv * r_comp) + A_ub_det.T @ (D * r_primal_ub))
        else:
            M = Q_det.clone()
            rhs_x = -r_dual.clone()

        if m_eq > 0:
            # Augmented system with equality constraints
            KKT_size = n + m_eq
            KKT_mat = torch.zeros(KKT_size, KKT_size, dtype=dtype, device=device)
            KKT_mat[:n, :n] = M
            KKT_mat[:n, n:] = A_eq_det.T
            KKT_mat[n:, :n] = A_eq_det

            rhs_full = torch.zeros(KKT_size, dtype=dtype, device=device)
            rhs_full[:n] = rhs_x
            rhs_full[n:] = -r_primal_eq

            reg = 1e-10 * torch.eye(KKT_size, dtype=dtype, device=device)
            sol = torch.linalg.solve(KKT_mat + reg, rhs_full)
            dx = sol[:n]
            dlam_eq = sol[n:]
        else:
            reg = 1e-10 * torch.eye(n, dtype=dtype, device=device)
            dx = torch.linalg.solve(M + reg, rhs_x)

        # Recover ds and dlam_ub
        if m_ub > 0:
            ds = -r_primal_ub - A_ub_det @ dx
            dlam_ub = S_inv * (-r_comp - lam_ub * ds)
        else:
            ds = torch.empty(0, dtype=dtype, device=device)
            dlam_ub = torch.empty(0, dtype=dtype, device=device)

        # Step size (fraction-to-boundary rule)
        alpha_p = 1.0
        alpha_d = 1.0
        tau = 0.995

        if m_ub > 0:
            neg_ds = ds < 0
            if neg_ds.any():
                alpha_p = min(alpha_p, (tau * (-s[neg_ds] / ds[neg_ds])).min().item())

            neg_dlam = dlam_ub < 0
            if neg_dlam.any():
                alpha_d = min(alpha_d, (tau * (-lam_ub[neg_dlam] / dlam_ub[neg_dlam])).min().item())

        # Update
        x = x + alpha_p * dx
        if m_ub > 0:
            s = s + alpha_p * ds
            lam_ub = lam_ub + alpha_d * dlam_ub
        if m_eq > 0:
            lam_eq = lam_eq + alpha_d * dlam_eq

    converged = torch.tensor(res_norm < tolerance, device=device)
    f_val = 0.5 * x @ Q_det @ x + c_det @ x

    A_ub_for_grad = A_ub if A_ub is not None else torch.empty(0, n, dtype=dtype, device=device)
    b_ub_for_grad = b_ub if b_ub is not None else torch.empty(0, dtype=dtype, device=device)
    A_eq_for_grad = A_eq if A_eq is not None else torch.empty(0, n, dtype=dtype, device=device)
    b_eq_for_grad = b_eq if b_eq is not None else torch.empty(0, dtype=dtype, device=device)

    x_with_grad = _QPImplicitGrad.apply(
        x, Q, c, A_ub_for_grad, b_ub_for_grad, A_eq_for_grad, b_eq_for_grad
    )

    return OptimizeResult(
        x=x_with_grad,
        converged=converged,
        num_iterations=torch.tensor(num_iter, dtype=torch.int64, device=device),
        fun=f_val,
    )
```

**Step 2: Run tests**

Run: `uv run pytest tests/torchscience/optimization/linear/test__quadratic_program.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/torchscience/optimization/linear/__init__.py src/torchscience/optimization/linear/_quadratic_program.py tests/torchscience/optimization/linear/test__quadratic_program.py
git commit -m "feat(optimization): add quadratic programming solver"
```

---

### Task 3: Linear Programming — Tests

**Files:**
- Create: `tests/torchscience/optimization/linear/test__linear_program.py`

**Step 1: Write tests**

```python
import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.linear._linear_program import linear_program


class TestLinearProgram:
    def test_simple_2d(self):
        """min -x1 - x2 s.t. x1 + x2 <= 4, x1 <= 3, x2 <= 3, x >= 0."""
        c = torch.tensor([-1.0, -1.0])
        A_ub = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        b_ub = torch.tensor([4.0, 3.0, 3.0])
        result = linear_program(
            c, A_ub=A_ub, b_ub=b_ub,
            bounds=(torch.zeros(2), None),
        )
        # Optimal: x = [3, 1] or [1, 3]; objective = -4
        assert result.fun.item() < -3.9

    def test_equality_constraint(self):
        """min c^T x s.t. A_eq x = b_eq, x >= 0."""
        c = torch.tensor([-1.0, -2.0])
        A_eq = torch.tensor([[1.0, 1.0]])
        b_eq = torch.tensor([1.0])
        result = linear_program(
            c, A_eq=A_eq, b_eq=b_eq,
            bounds=(torch.zeros(2), None),
        )
        # Optimal: x = [0, 1], objective = -2
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 1.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_result_type(self):
        """Test that result is an OptimizeResult."""
        c = torch.tensor([-1.0])
        A_ub = torch.tensor([[1.0]])
        b_ub = torch.tensor([5.0])
        result = linear_program(c, A_ub=A_ub, b_ub=b_ub, bounds=(torch.zeros(1), None))
        assert isinstance(result, OptimizeResult)

    def test_convergence_flag(self):
        """Test convergence flag."""
        c = torch.tensor([-1.0, -2.0])
        A_eq = torch.tensor([[1.0, 1.0]])
        b_eq = torch.tensor([1.0])
        result = linear_program(
            c, A_eq=A_eq, b_eq=b_eq,
            bounds=(torch.zeros(2), None),
        )
        assert result.converged.item() is True


class TestLinearProgramDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""
        c = torch.tensor([-1.0], dtype=dtype)
        A_ub = torch.tensor([[1.0]], dtype=dtype)
        b_ub = torch.tensor([5.0], dtype=dtype)
        result = linear_program(
            c, A_ub=A_ub, b_ub=b_ub,
            bounds=(torch.zeros(1, dtype=dtype), None),
        )
        assert result.x.dtype == dtype
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/torchscience/optimization/linear/test__linear_program.py -v 2>&1 | head -20`
Expected: ImportError

---

### Task 4: Linear Programming — Implementation

**Files:**
- Create: `src/torchscience/optimization/linear/_linear_program.py`
- Modify: `src/torchscience/optimization/linear/__init__.py`

**Step 1: Implement LP solver**

Convert LP to standard form and solve using primal-dual interior point (Mehrotra predictor-corrector). The LP can be expressed as a QP with `Q = 0`, but a dedicated LP solver is more numerically stable.

The implementation converts the LP to standard form: `min c^T x s.t. A x = b, x >= 0`, then applies the interior point method.

```python
from typing import Optional, Tuple

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult


def linear_program(
    c: Tensor,
    A_ub: Optional[Tensor] = None,
    b_ub: Optional[Tensor] = None,
    A_eq: Optional[Tensor] = None,
    b_eq: Optional[Tensor] = None,
    bounds: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None),
    *,
    maxiter: int = 1000,
    tolerance: float = 1e-8,
) -> OptimizeResult:
    r"""
    Linear programming via primal-dual interior point method.

    Solves: min c^T x
    subject to: A_ub @ x <= b_ub, A_eq @ x = b_eq, lower <= x <= upper

    Parameters
    ----------
    c : Tensor
        Cost vector of shape ``(n,)``.
    A_ub : Tensor, optional
        Inequality constraint matrix of shape ``(m_ub, n)``.
    b_ub : Tensor, optional
        Inequality constraint RHS of shape ``(m_ub,)``.
    A_eq : Tensor, optional
        Equality constraint matrix of shape ``(m_eq, n)``.
    b_eq : Tensor, optional
        Equality constraint RHS of shape ``(m_eq,)``.
    bounds : tuple, optional
        (lower, upper) bounds for x. Each is a Tensor of shape ``(n,)``
        or None for unbounded.
    maxiter : int, optional
        Maximum iterations. Default: 1000.
    tolerance : float, optional
        Convergence tolerance. Default: 1e-8.

    Returns
    -------
    OptimizeResult
        Result with ``x``, ``converged``, ``num_iterations``, ``fun``.
    """
    n = c.shape[0]
    dtype = c.dtype
    device = c.device
    lower, upper = bounds

    # Convert to standard form: min c_std^T x_std s.t. A_std x_std = b_std, x_std >= 0
    # 1. Handle bounds by variable substitution
    # 2. Convert inequalities to equalities with slack variables

    # Shift variables for lower bounds: x' = x - lower
    if lower is not None:
        shift = lower.detach()
    else:
        shift = torch.zeros(n, dtype=dtype, device=device)

    # Collect equality rows
    eq_rows = []
    eq_rhs = []

    if A_eq is not None and b_eq is not None:
        eq_rows.append(A_eq.detach())
        eq_rhs.append(b_eq.detach() - A_eq.detach() @ shift)

    # Convert inequalities: A_ub x <= b_ub  =>  A_ub x' + s = b_ub - A_ub shift, s >= 0
    n_slack = 0
    if A_ub is not None and b_ub is not None:
        m_ub = A_ub.shape[0]
        n_slack = m_ub
        A_ub_det = A_ub.detach()
        # [A_ub | I] [x'; s] = b_ub - A_ub shift
        A_with_slack = torch.cat([A_ub_det, torch.eye(m_ub, dtype=dtype, device=device)], dim=1)
        eq_rows.append(A_with_slack)
        eq_rhs.append(b_ub.detach() - A_ub_det @ shift)

    # Convert upper bounds: x' <= upper - lower => x' + s_upper = upper - lower
    n_upper_slack = 0
    if upper is not None:
        ub_effective = upper.detach() - shift
        n_upper_slack = n
        A_upper = torch.cat([torch.eye(n, dtype=dtype, device=device), torch.zeros(n, n_slack, dtype=dtype, device=device), torch.eye(n, dtype=dtype, device=device)], dim=1)
        eq_rows.append(A_upper)
        eq_rhs.append(ub_effective)

    n_total = n + n_slack + n_upper_slack

    if len(eq_rows) == 0:
        # No constraints — LP is unbounded unless bounded
        return OptimizeResult(
            x=shift,
            converged=torch.tensor(False, device=device),
            num_iterations=torch.tensor(0, dtype=torch.int64, device=device),
            fun=c.detach() @ shift,
        )

    A_std = torch.cat(eq_rows, dim=0)
    b_std = torch.cat(eq_rhs, dim=0)
    m_std = A_std.shape[0]

    # Pad cost vector for slack variables
    c_std = torch.zeros(n_total, dtype=dtype, device=device)
    c_std[:n] = c.detach()

    # Initialize primal and dual variables
    x_std = torch.ones(n_total, dtype=dtype, device=device)
    s = torch.ones(n_total, dtype=dtype, device=device)  # dual slacks
    lam = torch.zeros(m_std, dtype=dtype, device=device)  # equality duals

    num_iter = 0

    for k in range(maxiter):
        num_iter = k + 1

        # Residuals
        r_dual = c_std - A_std.T @ lam - s
        r_primal = A_std @ x_std - b_std
        mu = (x_std * s).sum() / n_total

        # Check convergence
        res_norm = max(r_dual.abs().max().item(), r_primal.abs().max().item(), mu.item())
        if res_norm < tolerance:
            break

        # Centering
        sigma = 0.3
        mu_target = sigma * mu

        # Newton system (reduced via elimination of ds)
        # X^{-1} S diagonal
        X_inv_S = s / x_std.clamp(min=1e-12)

        # (A D A^T) dlam = -(r_primal + A D (r_dual - X^{-1}(mu_target - X S)))
        D = 1.0 / X_inv_S.clamp(min=1e-12)
        r_comp = x_std * s - mu_target

        rhs_lam = -(r_primal + A_std @ (D * (r_dual - r_comp / x_std.clamp(min=1e-12))))

        ADA = A_std @ torch.diag(D) @ A_std.T
        reg = 1e-10 * torch.eye(m_std, dtype=dtype, device=device)

        try:
            dlam = torch.linalg.solve(ADA + reg, rhs_lam)
        except RuntimeError:
            break

        dx = D * (A_std.T @ dlam - r_dual + r_comp / x_std.clamp(min=1e-12))
        ds = -(r_comp + s * dx) / x_std.clamp(min=1e-12)

        # Step size (fraction-to-boundary)
        tau = 0.995
        alpha_p = 1.0
        alpha_d = 1.0

        neg_dx = dx < 0
        if neg_dx.any():
            alpha_p = min(alpha_p, (tau * (-x_std[neg_dx] / dx[neg_dx])).min().item())

        neg_ds = ds < 0
        if neg_ds.any():
            alpha_d = min(alpha_d, (tau * (-s[neg_ds] / ds[neg_ds])).min().item())

        x_std = x_std + alpha_p * dx
        lam = lam + alpha_d * dlam
        s = s + alpha_d * ds

    converged = torch.tensor(res_norm < tolerance, device=device)
    x_result = x_std[:n] + shift
    f_val = c.detach() @ x_result

    return OptimizeResult(
        x=x_result,
        converged=converged,
        num_iterations=torch.tensor(num_iter, dtype=torch.int64, device=device),
        fun=f_val,
    )
```

**Step 2: Update `__init__.py`**

Add `linear_program` to exports.

**Step 3: Run tests**

Run: `uv run pytest tests/torchscience/optimization/linear/test__linear_program.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/optimization/linear/_linear_program.py src/torchscience/optimization/linear/__init__.py tests/torchscience/optimization/linear/test__linear_program.py
git commit -m "feat(optimization): add linear programming solver"
```

---

### Task 5: Least Squares — Tests

**Files:**
- Create: `tests/torchscience/optimization/linear/test__least_squares.py`

**Step 1: Write tests**

```python
import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.linear._least_squares import least_squares


class TestLeastSquares:
    def test_linear_system(self):
        """Solve a linear least squares problem."""

        def residuals(x):
            # ||Ax - b||^2 where A = I, b = [1, 2]
            return x - torch.tensor([1.0, 2.0])

        result = least_squares(residuals, torch.zeros(2))
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 2.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_nonlinear(self):
        """Solve a nonlinear least squares problem."""

        def residuals(x):
            # Fit y = a * exp(b * t) to data
            t = torch.tensor([0.0, 1.0, 2.0, 3.0])
            y = torch.tensor([1.0, 2.7, 7.4, 20.1])
            return x[0] * torch.exp(x[1] * t) - y

        result = least_squares(residuals, torch.tensor([1.0, 0.5]))
        # Should converge to approximately a=1, b=1
        assert result.fun.item() < 0.1

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def residuals(x):
            return x - torch.tensor([1.0])

        result = least_squares(residuals, torch.zeros(1))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.fun is not None

    def test_bounded(self):
        """Test bounded least squares."""

        def residuals(x):
            return x - torch.tensor([5.0])

        result = least_squares(
            residuals,
            torch.tensor([0.0]),
            bounds=(torch.tensor([0.0]), torch.tensor([3.0])),
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([3.0]),
            atol=1e-2,
            rtol=1e-2,
        )


class TestLeastSquaresDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def residuals(x):
            return x

        x0 = torch.tensor([1.0], dtype=dtype)
        result = least_squares(residuals, x0)
        assert result.x.dtype == dtype
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/torchscience/optimization/linear/test__least_squares.py -v 2>&1 | head -20`
Expected: ImportError

---

### Task 6: Least Squares — Implementation

**Files:**
- Create: `src/torchscience/optimization/linear/_least_squares.py`
- Modify: `src/torchscience/optimization/linear/__init__.py`

**Step 1: Implement least_squares**

Wraps existing Levenberg-Marquardt for unbounded case. For bounded case, uses variable transformation.

```python
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
    bounds: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None),
    method: str = "levenberg-marquardt",
    maxiter: int = 100,
    tolerance: Optional[float] = None,
) -> OptimizeResult:
    r"""
    Nonlinear least squares solver.

    Minimizes: ``0.5 * ||fn(x)||^2`` with optional box bounds.

    Parameters
    ----------
    fn : Callable[[Tensor], Tensor]
        Residual function returning a tensor of residuals.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    bounds : tuple, optional
        (lower, upper) box bounds. Each is a Tensor of shape ``(n,)``
        or ``None`` for unbounded.
    method : str, optional
        Solver method. Default: ``"levenberg-marquardt"``.
    maxiter : int, optional
        Maximum iterations. Default: 100.
    tolerance : float, optional
        Convergence tolerance. Default: ``torch.finfo(x0.dtype).eps ** 0.5``.

    Returns
    -------
    OptimizeResult
        Result with ``x``, ``converged``, ``num_iterations``,
        ``fun`` (sum of squared residuals at solution).
    """
    if tolerance is None:
        tolerance = torch.finfo(x0.dtype).eps ** 0.5

    lower, upper = bounds
    has_bounds = lower is not None or upper is not None

    if has_bounds:
        # Variable transformation: x = lower + (upper - lower) * sigmoid(z)
        # This maps unconstrained z to bounded x
        lb = lower.detach() if lower is not None else torch.full_like(x0, -1e10)
        ub = upper.detach() if upper is not None else torch.full_like(x0, 1e10)

        def _to_internal(x):
            x_clipped = x.clamp(lb + 1e-8, ub - 1e-8)
            return torch.log((x_clipped - lb) / (ub - x_clipped))

        def _to_external(z):
            return lb + (ub - lb) * torch.sigmoid(z)

        z0 = _to_internal(x0.detach())

        def fn_internal(z):
            return fn(_to_external(z))

        z_opt = levenberg_marquardt(
            fn_internal, z0, tol=tolerance, maxiter=maxiter,
        )
        x_opt = _to_external(z_opt.detach())
    else:
        x_opt = levenberg_marquardt(fn, x0, tol=tolerance, maxiter=maxiter)

    with torch.no_grad():
        r_final = fn(x_opt if not has_bounds else x_opt.detach())
        f_val = 0.5 * (r_final**2).sum()
        grad_norm = torch.norm(r_final)
        converged = torch.tensor(grad_norm.item() < tolerance, device=x0.device)

    return OptimizeResult(
        x=x_opt,
        converged=converged,
        num_iterations=torch.tensor(maxiter, dtype=torch.int64, device=x0.device),
        fun=f_val,
    )
```

**Step 2: Update `__init__.py`**

```python
from ._least_squares import least_squares
from ._linear_program import linear_program
from ._quadratic_program import quadratic_program

__all__ = [
    "least_squares",
    "linear_program",
    "quadratic_program",
]
```

**Step 3: Run tests**

Run: `uv run pytest tests/torchscience/optimization/linear/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/optimization/linear/_least_squares.py src/torchscience/optimization/linear/__init__.py tests/torchscience/optimization/linear/test__least_squares.py
git commit -m "feat(optimization): add least squares solver"
```

---

### Task 7: Update Top-Level Exports

**Files:**
- Modify: `src/torchscience/optimization/__init__.py`

**Step 1: Add `linear` submodule to top-level exports**

Add `from torchscience.optimization import linear` alongside existing `optim` export.

**Step 2: Run full test suite**

Run: `uv run pytest tests/torchscience/optimization/ -v --tb=short`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/torchscience/optimization/__init__.py
git commit -m "feat(optimization): export linear submodule"
```

---

### Task 8: Final Verification

**Step 1: Run full optimization test suite**

Run: `uv run pytest tests/torchscience/optimization/ -v --tb=short`
Expected: All tests PASS

**Step 2: Run linting**

Run: `uv run pre-commit run --all-files`
Expected: All checks PASS

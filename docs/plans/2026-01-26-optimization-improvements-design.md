# Optimization Module Improvements Design

## Overview

Expand `torchscience.optimization` with new solvers, a linear optimization submodule, and additional test functions. All additions follow PyTorch-native design (tensors everywhere, batching first-class, torch.compile friendly) with no SciPy compatibility layer.

## Current State

The module currently provides:

- **Minimization**: L-BFGS, Levenberg-Marquardt, `minimize()` interface, `curve_fit()`
- **Constrained**: Augmented Lagrangian
- **Combinatorial**: Sinkhorn (optimal transport)
- **Test functions**: Rosenbrock (C++ kernel)
- **Optimizer API**: `torch.optim`-compatible LBFGS
- **Utilities**: Line search (Strong Wolfe, Armijo backtracking)
- **Result type**: `OptimizeResult(x, converged, num_iterations, fun)`

Root finding exists as a separate module (`torchscience.root_finding`) and is not in scope.

## Implementation Order

1. Unconstrained minimization (4 new solvers)
2. Constrained optimization (3 new solvers)
3. Linear algebra-based optimization (new submodule, 3 solvers)
4. Test functions (6 new functions with C++ kernels)

---

## 1. Unconstrained Minimization

### 1.1 Newton-CG (`newton_cg`)

Truncated Newton method using conjugate gradient to approximately solve the Newton system.

- **Hessian-vector products**: Uses `torch.func.jvp` to compute `H @ v` without forming the full Hessian.
- **CG truncation**: Inner CG loop terminates early on negative curvature (Steihaug-style) or when residual is below `tolerance * ||grad||`.
- **Line search**: Strong Wolfe (reusing `_strong_wolfe_line_search`).
- **Implicit differentiation**: One Newton correction step at the optimum (same pattern as L-BFGS).
- **Files**:
  - `src/torchscience/optimization/minimization/_newton_cg.py`
  - `tests/torchscience/optimization/minimization/test__newton_cg.py`

### 1.2 Trust-Region Newton (`trust_region`)

Trust-region method with Steihaug-CG subproblem solver.

- **Subproblem**: Minimizes the quadratic model within a trust-region ball using CG. Steps truncated at the trust-region boundary or at negative curvature.
- **Radius update**: Standard ratio-based — expand on good agreement, shrink on poor.
- **Hessian access**: `torch.func.jvp` (same as Newton-CG).
- **No line search**: Trust region controls step size directly.
- **Files**:
  - `src/torchscience/optimization/minimization/_trust_region.py`
  - `tests/torchscience/optimization/minimization/test__trust_region.py`

### 1.3 Nonlinear Conjugate Gradient (`conjugate_gradient`)

First-order method for large-scale problems where L-BFGS memory is too expensive.

- **Update formulas**: Polak-Ribiere+ (default, with automatic restart), Fletcher-Reeves, and Hestenes-Stiefel via `variant` parameter.
- **Line search**: Strong Wolfe (required for CG convergence guarantees).
- **Restart**: Automatic when conjugacy is lost (beta < 0 in PR+).
- **Files**:
  - `src/torchscience/optimization/minimization/_conjugate_gradient.py`
  - `tests/torchscience/optimization/minimization/test__conjugate_gradient.py`

### 1.4 Nelder-Mead (`nelder_mead`)

Derivative-free simplex method for non-differentiable or noisy objectives.

- **Operations**: Reflect, expand, contract, shrink.
- **Batch support**: Each batch element maintains its own simplex.
- **No implicit differentiation**: `OptimizeResult.x` is detached (method does not use gradients).
- **Files**:
  - `src/torchscience/optimization/minimization/_nelder_mead.py`
  - `tests/torchscience/optimization/minimization/test__nelder_mead.py`

### 1.5 Expanded `minimize()` Interface

Add method dispatching for `"newton-cg"`, `"trust-region"`, `"conjugate-gradient"`, and `"nelder-mead"` alongside existing `"l-bfgs"` and `"levenberg-marquardt"`.

---

## 2. Constrained Optimization

### 2.1 Interior Point (`interior_point`)

Primal-dual interior point method for general nonlinear constrained problems.

- **Problem**: `min f(x) s.t. h(x) = 0, g(x) <= 0`
- **Approach**: Log-barrier for inequalities, Newton steps on the KKT system. Barrier parameter `mu` decreases toward zero.
- **Linear algebra**: `torch.linalg.solve` on the reduced KKT system.
- **Implicit differentiation**: Custom `torch.autograd.Function` using KKT conditions at the solution.
- **Files**:
  - `src/torchscience/optimization/constrained/_interior_point.py`
  - `tests/torchscience/optimization/constrained/test__interior_point.py`

### 2.2 Box-Bounded L-BFGS (`l_bfgs_b`)

L-BFGS with simple bound constraints (`lower <= x <= upper`).

- **Algorithm**: L-BFGS-B — identifies active bounds, uses L-BFGS on free variables, projects gradient onto feasible set.
- **Cauchy point**: Generalized Cauchy point computation to identify the active set.
- **Reuses**: Existing line search and L-BFGS two-loop recursion.
- **Files**:
  - `src/torchscience/optimization/constrained/_l_bfgs_b.py`
  - `tests/torchscience/optimization/constrained/test__l_bfgs_b.py`

### 2.3 Sequential Quadratic Programming (`sqp`)

Newton-type method for general nonlinear constrained problems.

- **Subproblem**: Solves a QP approximation at each iteration (quadratic objective with linearized constraints).
- **QP solver**: Uses `quadratic_program()` from the `linear/` submodule.
- **Merit function**: L1 penalty with Armijo line search.
- **Hessian**: Damped BFGS approximation of the Lagrangian Hessian.
- **Dependency**: Requires `quadratic_program()` to be implemented first.
- **Files**:
  - `src/torchscience/optimization/constrained/_sqp.py`
  - `tests/torchscience/optimization/constrained/test__sqp.py`

---

## 3. Linear Algebra-Based Optimization (New Submodule)

New submodule: `torchscience/optimization/linear/`

### 3.1 Linear Programming (`linear_program`)

```python
def linear_program(
    c: Tensor,                                          # (n,) cost vector
    A_ub: Tensor | None = None,                         # (m1, n) inequality matrix
    b_ub: Tensor | None = None,                         # (m1,) inequality RHS
    A_eq: Tensor | None = None,                         # (m2, n) equality matrix
    b_eq: Tensor | None = None,                         # (m2,) equality RHS
    bounds: tuple[Tensor | None, Tensor | None] = (None, None),
    *,
    maxiter: int = 1000,
    tolerance: float = 1e-8,
) -> OptimizeResult:
```

- **Method**: Primal-dual interior point (Mehrotra predictor-corrector).
- **Batched**: `c` can be `(B, n)` for multiple LPs.
- **Implicit differentiation**: Via KKT conditions at the solution.
- **Files**:
  - `src/torchscience/optimization/linear/__init__.py`
  - `src/torchscience/optimization/linear/_linear_program.py`
  - `tests/torchscience/optimization/linear/test__linear_program.py`

### 3.2 Quadratic Programming (`quadratic_program`)

```python
def quadratic_program(
    Q: Tensor,                                          # (n, n) PSD Hessian
    c: Tensor,                                          # (n,) linear cost
    A_ub: Tensor | None = None,
    b_ub: Tensor | None = None,
    A_eq: Tensor | None = None,
    b_eq: Tensor | None = None,
    *,
    maxiter: int = 1000,
    tolerance: float = 1e-8,
) -> OptimizeResult:
```

- **Problem**: `min 0.5 x^T Q x + c^T x` subject to linear constraints.
- **Method**: Primal-dual interior point.
- **Dual use**: Standalone and as SQP subproblem solver.
- **Implicit differentiation**: Through KKT conditions.
- **Files**:
  - `src/torchscience/optimization/linear/_quadratic_program.py`
  - `tests/torchscience/optimization/linear/test__quadratic_program.py`

### 3.3 Least Squares (`least_squares`)

```python
def least_squares(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    bounds: tuple[Tensor | None, Tensor | None] = (None, None),
    method: str = "levenberg-marquardt",
    maxiter: int = 100,
    tolerance: float = 1e-8,
) -> OptimizeResult:
```

- **Problem**: `min ||fn(x)||^2` with optional box bounds.
- **Methods**: `"levenberg-marquardt"` (reuses existing LM) and `"trust-region-reflective"` (handles bounds via variable transformation).
- **Difference from `curve_fit`**: Lower-level, no data fitting assumptions.
- **Files**:
  - `src/torchscience/optimization/linear/_least_squares.py`
  - `tests/torchscience/optimization/linear/test__least_squares.py`

---

## 4. Test Functions

Six new optimization benchmark functions, each with C++ kernels following the Rosenbrock pattern.

| Function | Formula | Dims | Min | Min Location |
|----------|---------|------|-----|-------------|
| `ackley` | `-20·exp(-0.2·√(mean(x²))) - exp(mean(cos(2πx))) + 20 + e` | Any | 0 | origin |
| `rastrigin` | `10n + Σ(x² - 10·cos(2πx))` | Any | 0 | origin |
| `himmelblau` | `(x₁² + x₂ - 11)² + (x₁ + x₂² - 7)²` | 2 | 0 | 4 minima |
| `beale` | Three-term polynomial in x₁, x₂ | 2 | 0 | (3, 0.5) |
| `booth` | `(x₁ + 2x₂ - 7)² + (2x₁ + x₂ - 5)²` | 2 | 0 | (1, 3) |
| `sphere` | `Σ xᵢ²` | Any | 0 | origin |

### Per-Function Implementation

1. **C++ kernel** (`csrc/kernel/optimization/test_functions/<fn>.h`): Forward, backward, backward_backward with analytical gradients
2. **CPU backend** (`csrc/cpu/optimization/test_functions.h`): Vectorized via `at::vec::Vectorized<T>` for pointwise functions; custom loops for reduction-style (ackley, rastrigin, sphere)
3. **Schema** in `csrc/torchscience.cpp`
4. **Meta/autograd/autocast backends** following existing patterns
5. **Python wrapper** (`src/torchscience/optimization/test_function/_<fn>.py`)
6. **Tests** (`tests/torchscience/optimization/test_function/test__<fn>.py`): Correctness at known minima, gradcheck, gradgradcheck, meta tensors, autocast

Note: Ackley, Rastrigin, and Sphere are reductions over variable dimensions, not pointwise — they need custom C++ implementations rather than pointwise macros.

---

## Cross-Cutting Requirements

Every new solver must:

- Return `OptimizeResult(x, converged, num_iterations, fun)`
- Support implicit differentiation through the solution (except Nelder-Mead)
- Support batched inputs (first dimension is batch)
- Work with `torch.compile` (no graph breaks in hot loops)
- Have comprehensive tests:
  - Forward correctness against known solutions
  - `torch.autograd.gradcheck` (first-order)
  - `torch.autograd.gradgradcheck` (second-order)
  - Meta tensor shape inference
  - Autocast behavior
  - Edge cases (already converged, max iterations, singular systems)

## Module Structure (Final)

```
torchscience/optimization/
├── __init__.py
├── _result.py                    # OptimizeResult
├── _line_search.py               # Shared line search utilities
├── minimization/
│   ├── __init__.py
│   ├── _minimize.py              # Expanded dispatch
│   ├── _l_bfgs.py                # (existing)
│   ├── _levenberg_marquardt.py   # (existing)
│   ├── _curve_fit.py             # (existing)
│   ├── _newton_cg.py             # NEW
│   ├── _trust_region.py          # NEW
│   ├── _conjugate_gradient.py    # NEW
│   └── _nelder_mead.py           # NEW
├── constrained/
│   ├── __init__.py
│   ├── _augmented_lagrangian.py  # (existing)
│   ├── _interior_point.py        # NEW
│   ├── _l_bfgs_b.py              # NEW
│   └── _sqp.py                   # NEW
├── linear/                       # NEW submodule
│   ├── __init__.py
│   ├── _linear_program.py        # NEW
│   ├── _quadratic_program.py     # NEW
│   └── _least_squares.py         # NEW
├── combinatorial/
│   ├── __init__.py
│   └── _sinkhorn.py              # (existing)
├── test_function/
│   ├── __init__.py
│   ├── _rosenbrock.py            # (existing)
│   ├── _ackley.py                # NEW
│   ├── _rastrigin.py             # NEW
│   ├── _himmelblau.py            # NEW
│   ├── _beale.py                 # NEW
│   ├── _booth.py                 # NEW
│   └── _sphere.py                # NEW
└── optim/
    ├── __init__.py
    └── _lbfgs.py                 # (existing)
```

# Optimization Phase 4: Test Functions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add six optimization benchmark test functions (Ackley, Rastrigin, Himmelblau, Beale, Booth, Sphere) with C++ kernels, following the existing Rosenbrock pattern.

**Architecture:** Each function requires C++ kernel headers (forward, backward, backward_backward), CPU backend registration, meta/autograd/autocast backends, schema definitions, Python wrappers, and tests. The 2D functions (Himmelblau, Beale, Booth) are pointwise binary operators. The n-dimensional functions (Ackley, Rastrigin, Sphere) are reductions over the variable dimension.

**Tech Stack:** C++ (PyTorch ATen, `at::vec::Vectorized<T>` for SIMD), Python (torch.ops dispatching)

---

### Important Note on C++ Kernel Pattern

Each test function follows the Rosenbrock pattern discovered in the codebase:

1. **Kernel header** (`csrc/kernel/optimization/test_functions/<fn>.h`): Device-agnostic scalar implementation
2. **CPU backend** (`csrc/cpu/optimization/test_functions.h`): Registers vectorized CPU implementation using macros
3. **Schema** (`csrc/torchscience.cpp`): `TORCH_LIBRARY` schema definitions for forward, backward, backward_backward
4. **Meta backend** (`csrc/meta/`): Shape inference
5. **Autograd backend** (`csrc/autograd/`): Gradient wrappers
6. **Autocast backend** (`csrc/autocast/`): Mixed precision wrappers
7. **Python wrapper** (`src/torchscience/optimization/test_function/_<fn>.py`): Calls `torch.ops.torchscience.<fn>`
8. **Tests** (`tests/torchscience/optimization/test_function/test__<fn>.py`)

Due to the complexity of C++ kernel implementation (SIMD vectorization, macro registration, cmake integration), each task below covers a complete function end-to-end.

---

### Task 1: Sphere Function

The simplest test function. Good starting point to validate the pattern.

**Formula:** `f(x) = sum(x_i^2)` — minimum at origin, value 0.

**Files to create/modify:**
- `csrc/kernel/optimization/test_functions/sphere.h`
- `csrc/kernel/optimization/test_functions/sphere_backward.h`
- `csrc/kernel/optimization/test_functions/sphere_backward_backward.h`
- `csrc/cpu/optimization/test_functions.h` (add sphere entries)
- `csrc/torchscience.cpp` (add schema)
- `csrc/meta/optimization/test_functions.h` (add meta)
- `csrc/autograd/optimization/test_functions.h` (add autograd)
- `csrc/autocast/optimization/test_functions.h` (add autocast)
- `src/torchscience/optimization/test_function/_sphere.py`
- `src/torchscience/optimization/test_function/__init__.py` (add export)
- `tests/torchscience/optimization/test_function/test__sphere.py`

**Step 1: Study the Rosenbrock implementation to understand exact patterns**

Read these files to understand the exact C++ patterns:
- `csrc/kernel/optimization/test_functions/rosenbrock.h`
- `csrc/cpu/optimization/test_functions.h`
- `csrc/torchscience.cpp` (the rosenbrock entries)
- `csrc/meta/optimization/test_functions.h`
- `csrc/autograd/optimization/test_functions.h`
- `csrc/autocast/optimization/test_functions.h`
- `src/torchscience/optimization/test_function/_rosenbrock.py`
- `src/torchscience/optimization/test_function/__init__.py`

**Step 2: Implement the sphere function following the exact same patterns**

Note: Sphere is a **reduction** (sum over dimensions), not pointwise like Rosenbrock. This means it cannot use the `TORCHSCIENCE_CPU_POINTWISE_*` macros. Instead, implement a custom CPU kernel that iterates over elements and sums.

**Kernel forward (`sphere.h`):**
```cpp
template <typename T>
T sphere(T x) {
    return x * x;
}
// Note: sphere is actually a reduction f(x1,...,xn) = sum(xi^2)
// The forward kernel reduces across the last dimension
```

**Backward:** `df/dx_i = 2 * x_i`

**Backward_backward:** `d²f/dx_i² = 2` (constant Hessian)

**Python wrapper:**
```python
def sphere(x: Tensor) -> Tensor:
    r"""Sphere function."""
    return torch.ops.torchscience.sphere(x)
```

**Test file (`test__sphere.py`):**
```python
import pytest
import torch
import torch.testing

from torchscience.optimization.test_function import sphere


class TestSphere:
    def test_minimum_at_origin(self):
        x = torch.zeros(5)
        result = sphere(x)
        torch.testing.assert_close(result, torch.tensor(0.0), atol=1e-8, rtol=1e-8)

    def test_known_value(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = sphere(x)
        expected = torch.tensor(14.0)  # 1 + 4 + 9
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_gradient(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = sphere(x)
        result.backward()
        expected_grad = torch.tensor([2.0, 4.0, 6.0])
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-6, rtol=1e-6)

    def test_gradcheck(self):
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(sphere, (x,))

    def test_gradgradcheck(self):
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(sphere, (x,))

    def test_batch(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = sphere(x)
        expected = torch.tensor([5.0, 25.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_meta_shape(self):
        x = torch.empty(3, 5, device="meta")
        result = sphere(x)
        assert result.shape == (3,)
        assert result.device.type == "meta"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        x = torch.randn(5, dtype=dtype)
        result = sphere(x)
        assert result.dtype == dtype
```

**Step 3: Build and run tests**

Run: `uv sync && uv run pytest tests/torchscience/optimization/test_function/test__sphere.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(optimization): add sphere test function with C++ kernel"
```

---

### Task 2: Booth Function

**Formula:** `f(x1, x2) = (x1 + 2*x2 - 7)^2 + (2*x1 + x2 - 5)^2`
**Minimum:** f(1, 3) = 0. This is a 2D pointwise binary operator.

**Files:** Same pattern as Task 1 but for booth.

**Kernel:**
- Forward: `(x1 + 2*x2 - 7)^2 + (2*x1 + x2 - 5)^2`
- Backward w.r.t. x1: `2*(x1 + 2*x2 - 7) + 4*(2*x1 + x2 - 5)`
- Backward w.r.t. x2: `4*(x1 + 2*x2 - 7) + 2*(2*x1 + x2 - 5)`

**Test file should verify:**
- Minimum value at (1, 3)
- Known values at other points
- gradcheck and gradgradcheck
- Batch support
- Meta tensor shapes
- dtype preservation

**Step 1: Implement following Rosenbrock binary operator pattern**

Since Booth takes two scalar inputs (x1, x2), it can use the `TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR` macro.

**Step 2: Build and test**

Run: `uv sync && uv run pytest tests/torchscience/optimization/test_function/test__booth.py -v`

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(optimization): add Booth test function with C++ kernel"
```

---

### Task 3: Beale Function

**Formula:** `f(x1, x2) = (1.5 - x1 + x1*x2)^2 + (2.25 - x1 + x1*x2^2)^2 + (2.625 - x1 + x1*x2^3)^2`
**Minimum:** f(3, 0.5) = 0.

**Files:** Same pattern, binary operator.

**Kernel (analytical):**
- Forward: sum of three squared terms
- Backward: chain rule through each term
- Backward_backward: second derivatives

**Step 1: Implement**
**Step 2: Build and test**
**Step 3: Commit**

```bash
git commit -m "feat(optimization): add Beale test function with C++ kernel"
```

---

### Task 4: Himmelblau Function

**Formula:** `f(x1, x2) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2`
**Minima:** Four minima, all with f = 0:
- (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)

**Files:** Same pattern, binary operator.

**Step 1: Implement**
**Step 2: Build and test**
**Step 3: Commit**

```bash
git commit -m "feat(optimization): add Himmelblau test function with C++ kernel"
```

---

### Task 5: Rastrigin Function

**Formula:** `f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))`
**Minimum:** f(0,...,0) = 0. Highly multimodal (many local minima).

**Files:** Same pattern as sphere — reduction over variable dimension.

**Notes:** This is a reduction, not pointwise. Requires custom CPU kernel (not macro-based).

**Step 1: Implement**
**Step 2: Build and test**
**Step 3: Commit**

```bash
git commit -m "feat(optimization): add Rastrigin test function with C++ kernel"
```

---

### Task 6: Ackley Function

**Formula:** `f(x) = -20*exp(-0.2*sqrt(mean(x_i^2))) - exp(mean(cos(2*pi*x_i))) + 20 + e`
**Minimum:** f(0,...,0) = 0.

**Files:** Same pattern — reduction over variable dimension.

**Notes:** Most complex kernel due to exp, sqrt, cos, and mean operations. Requires careful numerical implementation to avoid overflow.

**Step 1: Implement**
**Step 2: Build and test**
**Step 3: Commit**

```bash
git commit -m "feat(optimization): add Ackley test function with C++ kernel"
```

---

### Task 7: Update Test Function Module Exports

**Files:**
- Modify: `src/torchscience/optimization/test_function/__init__.py`

**Step 1: Add all new exports**

```python
from ._ackley import ackley
from ._beale import beale
from ._booth import booth
from ._himmelblau import himmelblau
from ._rastrigin import rastrigin
from ._rosenbrock import rosenbrock
from ._sphere import sphere

__all__ = [
    "ackley",
    "beale",
    "booth",
    "himmelblau",
    "rastrigin",
    "rosenbrock",
    "sphere",
]
```

**Step 2: Run all test function tests**

Run: `uv run pytest tests/torchscience/optimization/test_function/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/torchscience/optimization/test_function/__init__.py
git commit -m "feat(optimization): export all test functions"
```

---

### Task 8: Final Verification

**Step 1: Run full optimization test suite**

Run: `uv run pytest tests/torchscience/optimization/ -v --tb=short`
Expected: All tests PASS

**Step 2: Run linting**

Run: `uv run pre-commit run --all-files`
Expected: All checks PASS

**Step 3: Verify all C++ builds**

Run: `uv sync`
Expected: Clean build with no warnings

"""Tests for torch.compile optimization utilities."""

import torch

from torchscience.integration import (
    euler,
    midpoint,
    runge_kutta_4,
)
from torchscience.integration._compile_utils import (
    compile_solver,
    is_compile_compatible,
)


class TestCompileCompatibility:
    """Test compile compatibility detection."""

    def test_fixed_step_solvers_compatible(self):
        """Fixed-step explicit solvers should be compile-compatible."""
        assert is_compile_compatible(euler)
        assert is_compile_compatible(midpoint)
        assert is_compile_compatible(runge_kutta_4)

    def test_compatibility_returns_bool(self):
        """is_compile_compatible should always return a boolean."""
        assert isinstance(is_compile_compatible(euler), bool)
        assert isinstance(is_compile_compatible(midpoint), bool)
        assert isinstance(is_compile_compatible(runge_kutta_4), bool)


class TestCompiledSolverCorrectness:
    """Test that compiled solvers produce correct results."""

    def test_compiled_euler_correctness(self):
        """Compiled Euler should produce same results as uncompiled."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Uncompiled
        y_ref, _ = euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        # Compiled
        compiled_euler = compile_solver(euler)
        y_compiled, _ = compiled_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert torch.allclose(y_ref, y_compiled, atol=1e-10)

    def test_compiled_midpoint_correctness(self):
        """Compiled midpoint should match uncompiled."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_ref, _ = midpoint(f, y0, t_span=(0.0, 1.0), dt=0.1)

        compiled_midpoint = compile_solver(midpoint)
        y_compiled, _ = compiled_midpoint(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert torch.allclose(y_ref, y_compiled, atol=1e-10)

    def test_compiled_rk4_correctness(self):
        """Compiled RK4 should match uncompiled."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_ref, _ = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        compiled_rk4 = compile_solver(runge_kutta_4)
        y_compiled, _ = compiled_rk4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert torch.allclose(y_ref, y_compiled, atol=1e-10)


class TestCompiledSolverGradients:
    """Test gradient computation through compiled solvers."""

    def test_gradient_through_compiled_euler(self):
        """Gradients should flow through compiled Euler."""
        k = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        compiled_euler = compile_solver(euler)
        y_final, _ = compiled_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()

    def test_gradient_through_compiled_rk4(self):
        """Gradients should flow through compiled RK4."""
        k = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        compiled_rk4 = compile_solver(runge_kutta_4)
        y_final, _ = compiled_rk4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()


class TestCompiledSolverMetadata:
    """Test that compiled solvers preserve useful metadata."""

    def test_compiled_solver_has_name(self):
        """Compiled solver should have informative name."""
        compiled_euler = compile_solver(euler)
        assert "euler" in compiled_euler.__name__
        assert "compiled" in compiled_euler.__name__

    def test_compiled_solver_has_docstring(self):
        """Compiled solver should have documentation."""
        compiled_euler = compile_solver(euler)
        assert compiled_euler.__doc__ is not None
        assert len(compiled_euler.__doc__) > 0


class TestCompiledSolverOptions:
    """Test compile_solver options."""

    def test_compile_with_mode(self):
        """Should accept compilation mode."""
        compiled = compile_solver(euler, mode="reduce-overhead")
        assert compiled is not None

    def test_compile_with_dynamic(self):
        """Should accept dynamic flag."""
        compiled = compile_solver(euler, dynamic=True)
        assert compiled is not None

    def test_compile_with_backend(self):
        """Should accept backend option."""
        compiled = compile_solver(euler, backend="eager")
        assert compiled is not None


class TestCompiledSolverHarmonicOscillator:
    """Test compiled solvers on harmonic oscillator."""

    def test_harmonic_oscillator_compiled_rk4(self):
        """Compiled RK4 should solve harmonic oscillator."""
        import math

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -q])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        compiled_rk4 = compile_solver(runge_kutta_4)
        y_final, _ = compiled_rk4(f, y0, t_span=(0.0, 2 * math.pi), dt=0.1)

        # After one period, should return close to initial
        assert torch.allclose(y_final, y0, atol=0.01)


class TestCompiledSolverInterpolant:
    """Test that compiled solvers return valid interpolants."""

    def test_compiled_euler_interpolant(self):
        """Compiled Euler should return valid interpolant."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        compiled_euler = compile_solver(euler)
        y_final, interp = compiled_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)

        # Interpolant should work
        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_compiled_rk4_interpolant_endpoints(self):
        """Compiled RK4 interpolant should match endpoints."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        compiled_rk4 = compile_solver(runge_kutta_4)
        y_final, interp = compiled_rk4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert torch.allclose(interp(0.0), y0, atol=1e-10)
        assert torch.allclose(interp(1.0), y_final, atol=1e-10)

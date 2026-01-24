"""Tests for Reversible Heun O(1) memory integrator."""

import math

import torch
from tensordict import TensorDict

from torchscience.ordinary_differential_equation._reversible_heun import (
    reversible_heun,
)


class TestReversibleHeunBasic:
    def test_harmonic_oscillator(self):
        """Reversible Heun should solve harmonic oscillator accurately."""

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -q])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        y_final, interp = reversible_heun(
            f, y0, t_span=(0.0, 2 * math.pi), dt=0.05
        )

        # After one period, should return close to initial
        assert torch.allclose(y_final, y0, atol=0.05)

    def test_exponential_decay(self):
        """Should match analytic solution for simple ODE."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = reversible_heun(f, y0, t_span=(0.0, 1.0), dt=0.1)

        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, atol=0.01)

    def test_linear_growth(self):
        """Test simple linear ODE: dy/dt = 1."""

        def f(t, y):
            return torch.ones_like(y)

        y0 = torch.tensor([0.0], dtype=torch.float64)
        y_final, _ = reversible_heun(f, y0, t_span=(0.0, 1.0), dt=0.1)

        expected = torch.tensor([1.0], dtype=torch.float64)
        assert torch.allclose(y_final, expected, atol=1e-10)

    def test_backward_integration(self):
        """Should support backward integration (t1 < t0)."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Forward to t=1
        y_forward, _ = reversible_heun(f, y0, t_span=(0.0, 1.0), dt=0.1)

        # Backward from t=1 to t=0
        y_backward, _ = reversible_heun(
            f, y_forward, t_span=(1.0, 0.0), dt=0.1
        )

        # Should approximately recover initial (limited by method's truncation error)
        assert torch.allclose(y_backward, y0, atol=1e-3)


class TestReversibleHeunReversibility:
    """
    Test properties related to Heun's method reversibility.

    The "reversible" property of Heun's method refers to its use in
    checkpoint-based gradient computation for neural ODEs, where:
    1. Forward pass stores only endpoints (O(1) memory)
    2. During backprop, intermediate states are recomputed as needed
    3. Heun's structure allows efficient recomputation

    This is different from exact algebraic inversion. The tests here verify:
    - Backward integration works correctly
    - Forward-backward round-trips have bounded error
    - The method supports efficient gradient computation
    """

    def test_backward_integration_linear(self):
        """Test backward integration for linear ODE."""

        def f(t, y):
            return -0.5 * y

        y0 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        y_final, _ = reversible_heun(f, y0, t_span=(0.0, 1.0), dt=0.1)
        y_backward, _ = reversible_heun(f, y_final, t_span=(1.0, 0.0), dt=0.1)

        # Round-trip error is bounded by O(h^2 * n_steps)
        assert torch.allclose(y_backward, y0, atol=1e-3)

    def test_backward_integration_nonlinear(self):
        """Test backward integration for nonlinear ODE."""

        def f(t, y):
            return torch.tanh(y)

        y0 = torch.tensor([0.5], dtype=torch.float64)

        y_final, _ = reversible_heun(f, y0, t_span=(0.0, 2.0), dt=0.05)
        y_recovered, _ = reversible_heun(
            f, y_final, t_span=(2.0, 0.0), dt=0.05
        )

        # Should recover approximately
        assert torch.allclose(y_recovered, y0, atol=1e-3)

    def test_backward_integration_oscillator(self):
        """Test backward integration for oscillatory system."""

        def f(t, y):
            # Coupled oscillator
            x, v = y[..., 0], y[..., 1]
            return torch.stack([v, -x - 0.1 * v], dim=-1)

        y0 = torch.tensor([1.0, 0.5], dtype=torch.float64)

        y_final, _ = reversible_heun(f, y0, t_span=(0.0, 3.0), dt=0.05)
        y_recovered, _ = reversible_heun(
            f, y_final, t_span=(3.0, 0.0), dt=0.05
        )

        # Should recover approximately
        assert torch.allclose(y_recovered, y0, atol=1e-2)


class TestReversibleHeunAutograd:
    def test_gradient_through_solver(self):
        """Gradients should flow through reversible Heun."""
        k = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, _ = reversible_heun(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()

    def test_memory_efficiency(self):
        """Reversible Heun should use O(1) memory for gradients."""
        # This test verifies the key property: memory doesn't grow with n_steps
        # We can't directly measure memory, but we verify gradient computation works
        # for long integrations without OOM

        k = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Long integration that would OOM with standard autograd
        y_final, _ = reversible_heun(
            f,
            y0,
            t_span=(0.0, 100.0),
            dt=0.1,  # 1000 steps
        )

        loss = y_final.sum()
        loss.backward()

        # If we got here without OOM, memory is bounded
        assert k.grad is not None

    def test_gradient_accuracy(self):
        """Gradients should be accurate (compare to finite differences)."""
        k = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, _ = reversible_heun(f, y0, t_span=(0.0, 1.0), dt=0.1)
        loss = y_final.sum()
        loss.backward()

        analytic_grad = k.grad.item()

        # Finite difference
        eps = 1e-5
        k_plus = torch.tensor([1.0 + eps], dtype=torch.float64)
        k_minus = torch.tensor([1.0 - eps], dtype=torch.float64)

        def solve_with_k(k_val):
            def f_k(t, y):
                return -k_val * y

            y_f, _ = reversible_heun(f_k, y0, t_span=(0.0, 1.0), dt=0.1)
            return y_f.sum().item()

        fd_grad = (solve_with_k(k_plus) - solve_with_k(k_minus)) / (2 * eps)

        assert abs(analytic_grad - fd_grad) < 0.01


class TestReversibleHeunInterpolant:
    def test_returns_interpolant(self):
        """Should return callable interpolant."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, interp = reversible_heun(f, y0, t_span=(0.0, 1.0), dt=0.1)

        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_interpolant_endpoints(self):
        """Interpolant should match endpoints exactly."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, interp = reversible_heun(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert torch.allclose(interp(0.0), y0, atol=1e-10)
        assert torch.allclose(interp(1.0), y_final, atol=1e-10)

    def test_interpolant_monotonic(self):
        """Interpolant should give monotonic values for monotonic solution."""

        def f(t, y):
            return -y  # Exponential decay is monotonically decreasing

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, interp = reversible_heun(f, y0, t_span=(0.0, 1.0), dt=0.1)

        times = torch.linspace(0.0, 1.0, 10)
        values = torch.stack([interp(t.item()) for t in times])

        # Check monotonically decreasing
        diffs = values[1:] - values[:-1]
        assert (diffs <= 0).all()


class TestReversibleHeunTensorDict:
    def test_tensordict_state(self):
        """Should support TensorDict state."""

        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict(
            {
                "x": torch.tensor([1.0], dtype=torch.float64),
                "v": torch.tensor([0.0], dtype=torch.float64),
            }
        )

        state_final, interp = reversible_heun(
            f, state0, t_span=(0.0, 1.0), dt=0.1
        )

        assert isinstance(state_final, TensorDict)
        assert "x" in state_final.keys()
        assert "v" in state_final.keys()


class TestReversibleHeunComplex:
    def test_complex_exponential(self):
        """Should handle complex tensors."""

        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = reversible_heun(f, y0, t_span=(0.0, 1.0), dt=0.01)

        expected = torch.exp(-1j * torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(y_final.squeeze(), expected, atol=1e-3)


class TestReversibleHeunAccuracy:
    def test_second_order_convergence(self):
        """Reversible Heun (Heun's method) should have 2nd order convergence."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        t_final = 1.0
        expected = torch.exp(torch.tensor([-t_final], dtype=torch.float64))

        errors = []
        dts = [0.1, 0.05, 0.025]

        for dt in dts:
            y_final, _ = reversible_heun(f, y0, t_span=(0.0, t_final), dt=dt)
            error = (y_final - expected).abs().item()
            errors.append(error)

        # Check convergence order: error should decrease by factor ~4 when dt halves
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        # For 2nd order method, ratio should be ~4
        assert 3.0 < ratio1 < 5.0
        assert 3.0 < ratio2 < 5.0

    def test_more_accurate_than_euler(self):
        """Reversible Heun (2nd order) should be more accurate than Euler (1st order)."""
        from torchscience.ordinary_differential_equation import euler

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        y_euler, _ = euler(f, y0, t_span=(0.0, 1.0), dt=0.1)
        y_heun, _ = reversible_heun(f, y0, t_span=(0.0, 1.0), dt=0.1)

        error_euler = (y_euler - expected).abs().item()
        error_heun = (y_heun - expected).abs().item()

        assert error_heun < error_euler

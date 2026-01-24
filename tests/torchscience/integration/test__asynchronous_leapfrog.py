"""Tests for Asynchronous Leapfrog (ALA) O(1) memory integrator."""

import math

import torch
from tensordict import TensorDict

from torchscience.integration._asynchronous_leapfrog import (
    asynchronous_leapfrog,
)


class TestAsynchronousLeapfrogBasic:
    """Basic functionality tests."""

    def test_harmonic_oscillator(self):
        """ALA should solve harmonic oscillator accurately."""

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -q])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        y_final, interp = asynchronous_leapfrog(
            f, y0, t_span=(0.0, 2 * math.pi), dt=0.05
        )

        # After one period, should return close to initial
        assert torch.allclose(y_final, y0, atol=0.1)

    def test_exponential_decay(self):
        """Should match analytic solution for simple ODE."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 1.0), dt=0.1)

        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, atol=0.05)

    def test_linear_growth(self):
        """Test simple linear ODE: dy/dt = 1."""

        def f(t, y):
            return torch.ones_like(y)

        y0 = torch.tensor([0.0], dtype=torch.float64)
        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 1.0), dt=0.1)

        expected = torch.tensor([1.0], dtype=torch.float64)
        assert torch.allclose(y_final, expected, atol=1e-10)


class TestAsynchronousLeapfrogReversibility:
    """Test the reversibility property of ALA."""

    def test_forward_backward_reversibility_linear(self):
        """ALA should recover initial state after forward-backward integration."""

        def f(t, y):
            return -0.5 * y

        y0 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        # Forward integration
        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 1.0), dt=0.1)

        # Backward integration should recover initial
        y_recovered, _ = asynchronous_leapfrog(
            f, y_final, t_span=(1.0, 0.0), dt=0.1
        )

        # Round-trip error should be bounded
        assert torch.allclose(y_recovered, y0, atol=1e-3)

    def test_forward_backward_reversibility_nonlinear(self):
        """Test reversibility for nonlinear ODE."""

        def f(t, y):
            return torch.tanh(y)

        y0 = torch.tensor([0.5], dtype=torch.float64)

        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 2.0), dt=0.05)
        y_recovered, _ = asynchronous_leapfrog(
            f, y_final, t_span=(2.0, 0.0), dt=0.05
        )

        # Should recover approximately
        assert torch.allclose(y_recovered, y0, atol=1e-2)

    def test_forward_backward_reversibility_oscillator(self):
        """Test reversibility for oscillatory system."""

        def f(t, y):
            x, v = y[..., 0], y[..., 1]
            return torch.stack([v, -x - 0.1 * v], dim=-1)

        y0 = torch.tensor([1.0, 0.5], dtype=torch.float64)

        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 3.0), dt=0.05)
        y_recovered, _ = asynchronous_leapfrog(
            f, y_final, t_span=(3.0, 0.0), dt=0.05
        )

        # Should recover approximately
        assert torch.allclose(y_recovered, y0, atol=0.05)


class TestAsynchronousLeapfrogAutograd:
    """Test gradient computation through ALA."""

    def test_gradient_through_solver(self):
        """Gradients should flow through ALA."""
        k = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()

    def test_gradient_accuracy(self):
        """Gradients should be accurate (compare to finite differences)."""
        k = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 1.0), dt=0.1)
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

            y_f, _ = asynchronous_leapfrog(f_k, y0, t_span=(0.0, 1.0), dt=0.1)
            return y_f.sum().item()

        fd_grad = (solve_with_k(k_plus) - solve_with_k(k_minus)) / (2 * eps)

        assert abs(analytic_grad - fd_grad) < 0.01


class TestAsynchronousLeapfrogLongIntegration:
    """Test long integrations without memory issues."""

    def test_long_integration_500_steps(self):
        """Should handle 500+ steps without memory issues."""
        k = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # 500 steps
        y_final, _ = asynchronous_leapfrog(
            f,
            y0,
            t_span=(0.0, 50.0),
            dt=0.1,
        )

        loss = y_final.sum()
        loss.backward()

        # If we got here without OOM, memory is bounded
        assert k.grad is not None

    def test_long_integration_1000_steps(self):
        """Should handle 1000 steps without memory issues."""
        k = torch.tensor([0.05], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # 1000 steps
        y_final, _ = asynchronous_leapfrog(
            f,
            y0,
            t_span=(0.0, 100.0),
            dt=0.1,
        )

        loss = y_final.sum()
        loss.backward()

        assert k.grad is not None


class TestAsynchronousLeapfrogTensorDict:
    """Test TensorDict support."""

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

        state_final, interp = asynchronous_leapfrog(
            f, state0, t_span=(0.0, 1.0), dt=0.1
        )

        assert isinstance(state_final, TensorDict)
        assert "x" in state_final.keys()
        assert "v" in state_final.keys()

    def test_tensordict_gradient(self):
        """Gradients should flow through TensorDict state."""
        k = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f(t, state):
            return TensorDict({"x": state["v"], "v": -k * state["x"]})

        state0 = TensorDict(
            {
                "x": torch.tensor([1.0], dtype=torch.float64),
                "v": torch.tensor([0.0], dtype=torch.float64),
            }
        )

        state_final, _ = asynchronous_leapfrog(
            f, state0, t_span=(0.0, 1.0), dt=0.1
        )

        loss = state_final["x"].sum() + state_final["v"].sum()
        loss.backward()

        assert k.grad is not None


class TestAsynchronousLeapfrogComplex:
    """Test complex tensor support."""

    def test_complex_exponential(self):
        """Should handle complex tensors."""

        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 1.0), dt=0.01)

        expected = torch.exp(-1j * torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(y_final.squeeze(), expected, atol=1e-3)

    def test_complex_oscillation(self):
        """Test complex valued oscillatory ODE."""

        def f(t, y):
            return 1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = asynchronous_leapfrog(
            f, y0, t_span=(0.0, math.pi), dt=0.01
        )

        # exp(i*pi) = -1
        expected = torch.tensor([-1.0 + 0j], dtype=torch.complex128)
        assert torch.allclose(y_final, expected, atol=0.01)


class TestAsynchronousLeapfrogBackward:
    """Test backward integration (t1 < t0)."""

    def test_backward_integration(self):
        """Should support backward integration (t1 < t0)."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Forward to t=1
        y_forward, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 1.0), dt=0.1)

        # Backward from t=1 to t=0
        y_backward, _ = asynchronous_leapfrog(
            f, y_forward, t_span=(1.0, 0.0), dt=0.1
        )

        # Should approximately recover initial
        assert torch.allclose(y_backward, y0, atol=1e-2)

    def test_backward_integration_direct(self):
        """Direct backward integration should give correct results."""

        def f(t, y):
            return -y

        # Analytic solution: y(t) = y0 * exp(-t)
        # At t=1, y(1) = exp(-1)
        # Going backward from t=1 to t=0: y(0) = y(1) * exp(1) = 1

        y1 = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        y_at_0, _ = asynchronous_leapfrog(f, y1, t_span=(1.0, 0.0), dt=0.1)

        expected = torch.tensor([1.0], dtype=torch.float64)
        assert torch.allclose(y_at_0, expected, atol=0.05)


class TestAsynchronousLeapfrogInterpolant:
    """Test interpolant functionality."""

    def test_returns_interpolant(self):
        """Should return callable interpolant."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, interp = asynchronous_leapfrog(
            f, y0, t_span=(0.0, 1.0), dt=0.1
        )

        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_interpolant_endpoints(self):
        """Interpolant should match endpoints exactly."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, interp = asynchronous_leapfrog(
            f, y0, t_span=(0.0, 1.0), dt=0.1
        )

        assert torch.allclose(interp(0.0), y0, atol=1e-10)
        assert torch.allclose(interp(1.0), y_final, atol=1e-10)

    def test_interpolant_monotonic(self):
        """Interpolant should give monotonic values for monotonic solution."""

        def f(t, y):
            return -y  # Exponential decay is monotonically decreasing

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, interp = asynchronous_leapfrog(
            f, y0, t_span=(0.0, 1.0), dt=0.1
        )

        times = torch.linspace(0.0, 1.0, 10)
        values = torch.stack([interp(t.item()) for t in times])

        # Check monotonically decreasing
        diffs = values[1:] - values[:-1]
        assert (diffs <= 0).all()


class TestAsynchronousLeapfrogConvergence:
    """Test convergence order."""

    def test_second_order_convergence(self):
        """ALA should have approximately 2nd order convergence."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        t_final = 1.0
        expected = torch.exp(torch.tensor([-t_final], dtype=torch.float64))

        errors = []
        dts = [0.1, 0.05, 0.025]

        for dt in dts:
            y_final, _ = asynchronous_leapfrog(
                f, y0, t_span=(0.0, t_final), dt=dt
            )
            error = (y_final - expected).abs().item()
            errors.append(error)

        # Check convergence order: error should decrease by factor ~4 when dt halves
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        # For 2nd order method, ratio should be ~4
        # Allow some tolerance since ALA has unique staggered structure
        assert 2.5 < ratio1 < 6.0
        assert 2.5 < ratio2 < 6.0

    def test_better_than_euler(self):
        """ALA should be more accurate than Euler (1st order)."""
        from torchscience.integration import euler

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        y_euler, _ = euler(f, y0, t_span=(0.0, 1.0), dt=0.1)
        y_ala, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 1.0), dt=0.1)

        error_euler = (y_euler - expected).abs().item()
        error_ala = (y_ala - expected).abs().item()

        assert error_ala < error_euler


class TestAsynchronousLeapfrogEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_step(self):
        """Should handle single step integration."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 0.1), dt=0.1)

        # After one step with dt=0.1
        assert y_final.shape == y0.shape
        assert not torch.isnan(y_final).any()

    def test_zero_time_span(self):
        """Should handle t0 == t1."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 0.0), dt=0.1)

        # Should return initial state unchanged
        assert torch.allclose(y_final, y0)

    def test_batched_state(self):
        """Should handle batched initial conditions."""

        def f(t, y):
            return -y

        # Batch of 5 initial conditions, each with 3 components
        y0 = torch.randn(5, 3, dtype=torch.float64)
        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert y_final.shape == y0.shape

    def test_high_dimensional_state(self):
        """Should handle high-dimensional state vectors."""

        def f(t, y):
            return -0.1 * y

        y0 = torch.randn(100, dtype=torch.float64)
        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert y_final.shape == y0.shape
        assert not torch.isnan(y_final).any()

"""Tests for Adams-Bashforth-Moulton multistep solver."""

import math

import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem._adams import adams


class TestAdamsBasic:
    """Basic functionality tests."""

    def test_exponential_decay(self):
        """Should solve simple exponential decay."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = adams(f, y0, t_span=(0.0, 1.0))

        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, atol=1e-5)

    def test_harmonic_oscillator(self):
        """Should solve harmonic oscillator."""

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -q])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        y_final, interp = adams(f, y0, t_span=(0.0, 2 * math.pi))

        # After one period, should return close to initial
        assert torch.allclose(y_final, y0, atol=0.01)

    def test_returns_interpolant(self):
        """Should return callable interpolant."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = adams(f, y0, t_span=(0.0, 1.0))

        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_linear_growth(self):
        """Should solve linear growth ODE."""

        def f(t, y):
            return y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = adams(f, y0, t_span=(0.0, 1.0))

        expected = torch.exp(torch.tensor([1.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, atol=1e-5)


class TestAdamsOrders:
    """Test different orders."""

    def test_order_1(self):
        """Order 1 should work (Adams-Bashforth 1 = Euler)."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = adams(f, y0, t_span=(0.0, 1.0), max_order=1)

        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        # Lower order = lower accuracy
        assert torch.allclose(y_final, expected, atol=0.1)

    def test_higher_order_more_accurate(self):
        """Higher orders should be more accurate at same tolerance."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        # Use loose tolerance so adaptive stepping doesn't dominate
        y_low, _ = adams(
            f, y0, t_span=(0.0, 1.0), max_order=1, rtol=1e-3, atol=1e-5
        )
        y_high, _ = adams(
            f, y0, t_span=(0.0, 1.0), max_order=5, rtol=1e-3, atol=1e-5
        )

        error_low = (y_low - exact).abs().item()
        error_high = (y_high - exact).abs().item()

        # Higher order should be at least as accurate
        assert error_high <= error_low * 1.1  # Allow small tolerance


class TestAdamsAdaptive:
    """Test adaptive step size control."""

    def test_respects_tolerances(self):
        """Solution should satisfy requested tolerances."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        rtol, atol = 1e-6, 1e-8
        y_final, _ = adams(f, y0, t_span=(0.0, 1.0), rtol=rtol, atol=atol)

        error = (y_final - exact).abs().item()
        assert error < 100 * rtol


class TestAdamsAutograd:
    """Test gradient computation."""

    def test_gradient_through_solver(self):
        """Gradients should flow through Adams."""
        k = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = adams(f, y0, t_span=(0.0, 1.0))

        loss = y_final.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()


class TestAdamsTensorDict:
    """Test TensorDict support."""

    def test_tensordict_state(self):
        """Should work with TensorDict state."""

        def f(t, state):
            return TensorDict(
                {
                    "x": state["v"],
                    "v": -state["x"],
                }
            )

        state0 = TensorDict(
            {
                "x": torch.tensor([1.0], dtype=torch.float64),
                "v": torch.tensor([0.0], dtype=torch.float64),
            }
        )

        state_final, interp = adams(f, state0, t_span=(0.0, 1.0))

        assert isinstance(state_final, TensorDict)
        assert "x" in state_final.keys()


class TestAdamsEdgeCases:
    """Test edge cases."""

    def test_backward_integration(self):
        """Should handle t1 < t0."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = adams(f, y0, t_span=(1.0, 0.0))

        expected = torch.exp(torch.tensor([1.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, atol=0.1)

    def test_zero_time_span(self):
        """Should handle t0 == t1."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = adams(f, y0, t_span=(0.0, 0.0))

        assert torch.allclose(y_final, y0)

    def test_batched_state(self):
        """Should handle batched initial conditions."""

        def f(t, y):
            return -y

        y0 = torch.randn(5, 3, dtype=torch.float64)
        y_final, _ = adams(f, y0, t_span=(0.0, 1.0))

        assert y_final.shape == y0.shape


class TestAdamsInterpolant:
    """Test interpolant functionality."""

    def test_interpolant_endpoints(self):
        """Interpolant should match endpoints."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = adams(f, y0, t_span=(0.0, 1.0))

        assert torch.allclose(interp(0.0), y0, atol=1e-10)
        assert torch.allclose(interp(1.0), y_final, atol=1e-10)

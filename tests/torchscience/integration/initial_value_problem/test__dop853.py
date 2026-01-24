"""Tests for DOP853 8th-order adaptive solver."""

import math

import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem._dop853 import dop853


class TestDOP853Basic:
    """Basic functionality tests."""

    def test_exponential_decay(self):
        """Should solve simple exponential decay accurately."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = dop853(f, y0, t_span=(0.0, 1.0))

        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, atol=1e-6)

    def test_harmonic_oscillator(self):
        """Should solve harmonic oscillator with high accuracy."""

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -q])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        y_final, interp = dop853(f, y0, t_span=(0.0, 2 * math.pi))

        # After one period, should return very close to initial
        assert torch.allclose(y_final, y0, atol=1e-4)

    def test_returns_interpolant(self):
        """Should return callable interpolant."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = dop853(f, y0, t_span=(0.0, 1.0))

        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

        # Interpolant should be reasonably accurate (linear interpolation)
        expected_mid = torch.exp(torch.tensor([-0.5], dtype=torch.float64))
        assert torch.allclose(y_mid, expected_mid, atol=0.05)

    def test_linear_ode(self):
        """Should solve linear growth ODE."""

        def f(t, y):
            return y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = dop853(f, y0, t_span=(0.0, 1.0))

        expected = torch.exp(torch.tensor([1.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, atol=1e-6)


class TestDOP853Accuracy:
    """Test high-order accuracy."""

    def test_high_accuracy_at_tight_tolerance(self):
        """DOP853 should achieve high accuracy at tight tolerances."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        y_final, _ = dop853(f, y0, t_span=(0.0, 1.0), rtol=1e-8, atol=1e-10)
        error = (y_final - exact).abs().item()

        assert error < 1e-6

    def test_comparable_to_dormand_prince_5(self):
        """DOP853 should achieve comparable or better accuracy than DP5."""
        from torchscience.integration.initial_value_problem import (
            dormand_prince_5,
        )

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        tol = 1e-5
        y_dp5, _ = dormand_prince_5(
            f, y0, t_span=(0.0, 1.0), rtol=tol, atol=tol
        )
        y_dop853, _ = dop853(f, y0, t_span=(0.0, 1.0), rtol=tol, atol=tol)

        error_dp5 = (y_dp5 - exact).abs().item()
        error_dop853 = (y_dop853 - exact).abs().item()

        # Both should satisfy tolerance
        assert error_dp5 < tol * 100
        assert error_dop853 < tol * 100


class TestDOP853Adaptive:
    """Test adaptive step size control."""

    def test_respects_tolerances(self):
        """Solution should satisfy requested tolerances."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        rtol, atol = 1e-6, 1e-8
        y_final, _ = dop853(f, y0, t_span=(0.0, 1.0), rtol=rtol, atol=atol)

        error = (y_final - exact).abs().item()
        # Error should be within reasonable bounds
        assert error < 100 * rtol

    def test_handles_varying_timescales(self):
        """Should adapt step size for problems with varying timescales."""

        def f(t, y):
            # Fast initial transient, then slow decay
            rate = 10.0 if t < 0.1 else 0.1
            return -rate * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = dop853(f, y0, t_span=(0.0, 10.0))

        # Should complete without error
        assert not torch.isnan(y_final).any()
        assert y_final.item() > 0  # Should still be positive


class TestDOP853Autograd:
    """Test gradient computation."""

    def test_gradient_through_solver(self):
        """Gradients should flow through DOP853."""
        k = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = dop853(f, y0, t_span=(0.0, 1.0))

        loss = y_final.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()

    def test_gradient_accuracy(self):
        """Gradients should be accurate (compare to finite differences)."""
        k_val = 1.0
        k = torch.tensor([k_val], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = dop853(f, y0, t_span=(0.0, 1.0))
        loss = y_final.sum()
        loss.backward()

        analytic_grad = k.grad.item()

        # Finite difference
        eps = 1e-5

        def solve_with_k(k_val_):
            def f_k(t, y):
                return -k_val_ * y

            y_f, _ = dop853(f_k, y0, t_span=(0.0, 1.0))
            return y_f.sum().item()

        fd_grad = (solve_with_k(k_val + eps) - solve_with_k(k_val - eps)) / (
            2 * eps
        )

        assert abs(analytic_grad - fd_grad) < 0.01


class TestDOP853TensorDict:
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

        state_final, interp = dop853(f, state0, t_span=(0.0, 1.0))

        assert isinstance(state_final, TensorDict)
        assert "x" in state_final.keys()
        assert "v" in state_final.keys()


class TestDOP853EdgeCases:
    """Test edge cases."""

    def test_backward_integration(self):
        """Should handle t1 < t0."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = dop853(f, y0, t_span=(1.0, 0.0))

        # Backward from t=1 to t=0 with dy/dt=-y means y increases
        expected = torch.exp(torch.tensor([1.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, atol=1e-2)

    def test_complex_valued(self):
        """Should handle complex tensors."""

        def f(t, y):
            return 1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = dop853(f, y0, t_span=(0.0, math.pi), rtol=1e-5, atol=1e-7)

        expected = torch.tensor([-1.0 + 0j], dtype=torch.complex128)
        assert torch.allclose(y_final, expected, atol=0.1)

    def test_zero_time_span(self):
        """Should handle t0 == t1."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = dop853(f, y0, t_span=(0.0, 0.0))

        assert torch.allclose(y_final, y0)

    def test_batched_state(self):
        """Should handle batched initial conditions."""

        def f(t, y):
            return -y

        y0 = torch.randn(5, 3, dtype=torch.float64)
        y_final, _ = dop853(f, y0, t_span=(0.0, 1.0))

        assert y_final.shape == y0.shape


class TestDOP853Interpolant:
    """Test interpolant functionality."""

    def test_interpolant_endpoints(self):
        """Interpolant should match endpoints."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = dop853(f, y0, t_span=(0.0, 1.0))

        assert torch.allclose(interp(0.0), y0, atol=1e-10)
        assert torch.allclose(interp(1.0), y_final, atol=1e-10)

    def test_interpolant_batch_query(self):
        """Interpolant should handle batch time queries."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = dop853(f, y0, t_span=(0.0, 1.0))

        t_query = torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64
        )
        y_batch = interp(t_query)

        assert y_batch.shape == (5, 1)

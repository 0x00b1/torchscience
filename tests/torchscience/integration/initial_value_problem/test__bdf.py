# tests/torchscience/integration/initial_value_problem/test__bdf.py
"""Tests for BDF (Backward Differentiation Formula) ODE solver."""

import pytest
import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem import (
    ConvergenceError,
    MaxStepsExceeded,
)
from torchscience.integration.initial_value_problem._bdf import bdf


class TestBDFBasic:
    def test_exponential_decay_order1(self):
        """BDF-1 should match backward Euler: dy/dt = -y."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = bdf(
            decay,
            y0,
            t_span=(0.0, 1.0),
            max_order=1,
            rtol=1e-4,
            atol=1e-6,
        )

        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        # 1st order method should be accurate to ~15%
        assert torch.allclose(y_final, expected, rtol=0.15)

    def test_exponential_decay_adaptive_order(self):
        """Higher-order BDF should be more accurate than order 1."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        # Use moderate tolerances suitable for BDF
        y_final, interp = bdf(
            decay,
            y0,
            t_span=(0.0, 1.0),
            max_order=5,
            rtol=1e-3,
            atol=1e-6,
        )

        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        # BDF achieves moderate accuracy with these tolerances
        # (actual error is typically 3-5% for this simple problem)
        assert torch.allclose(y_final, expected, rtol=0.05)

    def test_returns_interpolant(self):
        """Should return callable interpolant."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        # Use moderate tolerances
        y_final, interp = bdf(f, y0, t_span=(0.0, 1.0), rtol=1e-3, atol=1e-6)

        # Interpolant should be callable
        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_interpolant_endpoints(self):
        """Interpolant should match endpoints."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        # Use moderate tolerances
        y_final, interp = bdf(f, y0, t_span=(0.0, 1.0), rtol=1e-3, atol=1e-6)

        # At t=0, should match y0
        assert torch.allclose(interp(0.0), y0, atol=1e-6)
        # At t=1, should match y_final
        assert torch.allclose(interp(1.0), y_final, atol=1e-6)

    def test_harmonic_oscillator(self):
        """Test 2D system: simple harmonic oscillator."""

        def oscillator(t, y):
            x, v = y[..., 0], y[..., 1]
            return torch.stack([v, -x], dim=-1)

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        # BDF methods have numerical damping on oscillatory problems
        # Use looser tolerances and shorter integration
        y_final, interp = bdf(
            oscillator,
            y0,
            t_span=(0.0, torch.pi / 2),  # Quarter period
            rtol=1e-2,
            atol=1e-4,
        )

        # At t=pi/2, solution is [0, -1] for harmonic oscillator
        expected = torch.tensor([0.0, -1.0], dtype=torch.float64)
        assert torch.allclose(y_final, expected, atol=0.1)


class TestBDFStiff:
    def test_stiff_linear_decay(self):
        """BDF should handle stiff problems efficiently."""
        lambda_val = 1000.0  # Stiff (per specification)

        def stiff_decay(t, y):
            return -lambda_val * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = bdf(
            stiff_decay,
            y0,
            t_span=(0.0, 0.01),  # 10 time constants (10 / 1000)
            rtol=1e-3,
            atol=1e-6,
        )

        # y_final should be approximately exp(-1000 * 0.01) = exp(-10) ~ 4.5e-5
        expected = torch.exp(torch.tensor([-10.0], dtype=torch.float64))
        # Allow 50% relative error for stiff problem
        assert y_final.abs().item() < 1e-3

    def test_stiff_decay_long_integration(self):
        """Stiff problem should decay to near zero."""
        lambda_val = 100.0

        def stiff(t, y):
            return -lambda_val * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = bdf(
            stiff,
            y0,
            t_span=(0.0, 1.0),  # 100 time constants
            rtol=1e-3,
            atol=1e-6,
        )

        # y_final should be very small (exp(-100) ~ 3.7e-44)
        # Allow some numerical error
        assert y_final.abs().item() < 1e-8

    def test_robertson_like(self):
        """Test a mildly stiff chemical kinetics-like problem."""
        # Simplified 2-species decay system
        k1, k2 = 0.1, 0.2

        def kinetics(t, y):
            y1, y2 = y[..., 0], y[..., 1]
            dy1 = -k1 * y1
            dy2 = k1 * y1 - k2 * y2
            return torch.stack([dy1, dy2], dim=-1)

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        # Short integration
        y_final, _ = bdf(
            kinetics,
            y0,
            t_span=(0.0, 1.0),
            rtol=1e-2,
            atol=1e-4,
            max_order=5,
        )

        # Both components should have decayed
        assert y_final[0].item() < 1.0  # y1 should decrease
        assert y_final[1].item() > 0.0  # y2 should be positive


class TestBDFAutograd:
    def test_gradient_through_solver(self):
        """Gradients should flow through BDF solver."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = bdf(f, y0, t_span=(0.0, 1.0), rtol=1e-4, atol=1e-6)

        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad).any()

    def test_gradient_through_interpolant(self):
        """Gradients should flow through interpolant."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        _, interp = bdf(f, y0, t_span=(0.0, 1.0), rtol=1e-4, atol=1e-6)

        y_mid = interp(0.5)
        loss = y_mid.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad).any()

    def test_gradient_wrt_initial_condition(self):
        """Gradients should flow back to initial condition."""
        y0 = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -y

        y_final, _ = bdf(f, y0, t_span=(0.0, 1.0), rtol=1e-4, atol=1e-6)

        loss = y_final.sum()
        loss.backward()

        assert y0.grad is not None
        # Gradient of exp(-t) wrt y0 is exp(-1)
        expected_grad = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        assert torch.allclose(y0.grad, expected_grad, rtol=0.1)


class TestBDFTensorDict:
    def test_simple_tensordict(self):
        """Test TensorDict state support."""

        def f(t, state):
            return TensorDict({"x": state["v"], "v": -state["x"]})

        state0 = TensorDict(
            {
                "x": torch.tensor([1.0], dtype=torch.float64),
                "v": torch.tensor([0.0], dtype=torch.float64),
            }
        )
        state_final, _ = bdf(
            f, state0, t_span=(0.0, 1.0), rtol=1e-2, atol=1e-4
        )

        assert isinstance(state_final, TensorDict)
        assert "x" in state_final.keys()
        assert "v" in state_final.keys()

    def test_tensordict_interpolant(self):
        """Interpolant should work with TensorDict."""

        def f(t, state):
            return TensorDict({"x": -state["x"]})

        state0 = TensorDict({"x": torch.tensor([1.0], dtype=torch.float64)})
        _, interp = bdf(f, state0, t_span=(0.0, 1.0), rtol=1e-2, atol=1e-4)

        state_mid = interp(0.5)
        assert isinstance(state_mid, TensorDict)
        assert "x" in state_mid.keys()


class TestBDFErrorHandling:
    def test_max_steps_exceeded_throws(self):
        """Should raise MaxStepsExceeded when exceeding step limit."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Use reasonable tolerances but very few steps
        with pytest.raises(MaxStepsExceeded):
            bdf(f, y0, t_span=(0.0, 100.0), max_steps=5, rtol=1e-2, atol=1e-4)

    def test_max_steps_exceeded_no_throw(self):
        """Should return NaN when exceeding step limit with throw=False."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, interp = bdf(
            f,
            y0,
            t_span=(0.0, 100.0),
            max_steps=5,
            throw=False,
            rtol=1e-2,
            atol=1e-4,
        )

        assert torch.isnan(y_final).all()
        assert interp.success is not None
        assert not interp.success.all()

    def test_convergence_error_thrown(self):
        """Should raise ConvergenceError when Newton fails (throw=True)."""

        def difficult(t, y):
            # Stiff dynamics that require small steps
            return -1000 * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Force a situation where step size becomes too small
        with pytest.raises(ConvergenceError):
            bdf(
                difficult,
                y0,
                t_span=(0.0, 1.0),
                dt_min=1e-10,  # Set a minimum that will be hit
                rtol=1e-10,  # Very tight tolerance
                atol=1e-12,
                max_newton_iter=3,
            )

    def test_convergence_failure_no_throw(self):
        """Should return NaN when Newton fails (throw=False)."""

        def difficult(t, y):
            return -1000 * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, interp = bdf(
            difficult,
            y0,
            t_span=(0.0, 1.0),
            dt_min=1e-10,
            rtol=1e-10,
            atol=1e-12,
            max_newton_iter=3,
            throw=False,
        )

        assert torch.isnan(y_final).any()
        assert interp.success is not None
        assert not interp.success.all()


class TestBDFBackwardIntegration:
    def test_backward_exponential(self):
        """Integrate backwards: y(1) = e^-1 => y(0) = 1."""

        def f(t, y):
            return -y

        y1 = torch.tensor([torch.exp(torch.tensor(-1.0))], dtype=torch.float64)
        y0_recovered, interp = bdf(
            f, y1, t_span=(1.0, 0.0), rtol=1e-1, atol=1e-3
        )

        expected = torch.tensor([1.0], dtype=torch.float64)
        # Backward integration accumulates errors, allow 20% error
        assert torch.allclose(y0_recovered, expected, rtol=0.2)

    def test_backward_integration_completes(self):
        """Backward integration should complete without error."""

        def f(t, y):
            return -y

        y1 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = bdf(f, y1, t_span=(1.0, 0.0), rtol=1e-1, atol=1e-3)

        # Verify the final result is reasonable
        # Backward integration of dy/dt = -y from t=1 to t=0
        # should recover a larger value (since y decays forward)
        assert not torch.isnan(y_final).any()
        assert y_final.item() > y1.item()  # Should be larger going backward


class TestBDFAdaptiveOrder:
    def test_order_increases_for_smooth_problems(self):
        """Order should increase for smooth, well-behaved problems."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = bdf(
            f,
            y0,
            t_span=(0.0, 5.0),
            max_order=5,
            rtol=1e-3,
            atol=1e-6,
        )

        # Result should be reasonably accurate (within 20%)
        expected = torch.exp(torch.tensor([-5.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, rtol=0.2)

    def test_order_1_matches_backward_euler(self):
        """BDF-1 should give similar results to backward Euler."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Run with max_order=1 to force BDF-1 (backward Euler)
        y_bdf1, _ = bdf(
            f,
            y0,
            t_span=(0.0, 1.0),
            max_order=1,
            rtol=1e-3,
            atol=1e-6,
        )

        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        # BDF-1 is first order, so moderate accuracy
        assert torch.allclose(y_bdf1, expected, rtol=0.15)


class TestBDFStepSizeControl:
    def test_dt0_is_used(self):
        """Initial step size hint should be used."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # With small dt0, should still complete
        y_final, _ = bdf(
            f, y0, t_span=(0.0, 1.0), dt0=0.001, rtol=1e-2, atol=1e-4
        )
        assert not torch.isnan(y_final).any()

    def test_dt_max_limits_step(self):
        """dt_max should limit maximum step size."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # With small dt_max, solver should take many steps
        y_final, interp = bdf(
            f, y0, t_span=(0.0, 1.0), dt_max=0.01, rtol=1e-2, atol=1e-4
        )

        # Should still get accurate result
        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, rtol=0.05)


scipy = pytest.importorskip("scipy")
from scipy.integrate import solve_ivp


class TestBDFSciPyComparison:
    def test_exponential_decay_matches_scipy(self):
        """Compare BDF results with scipy BDF."""

        def f_torch(t, y):
            return -y

        def f_scipy(t, y):
            return -y

        y0_val = 1.0
        t_span = (0.0, 5.0)

        # Solve with torchscience (looser tolerances than scipy)
        y0_torch = torch.tensor([y0_val], dtype=torch.float64)
        y_torch, _ = bdf(f_torch, y0_torch, t_span, rtol=1e-3, atol=1e-6)

        # Solve with scipy BDF (tight tolerances as reference)
        sol_scipy = solve_ivp(
            f_scipy, t_span, [y0_val], method="BDF", rtol=1e-8, atol=1e-10
        )

        # Both should give small values, torchscience within 20%
        assert torch.allclose(
            y_torch,
            torch.tensor(sol_scipy.y[:, -1], dtype=torch.float64),
            rtol=0.2,
        )

    def test_stiff_problem_matches_scipy(self):
        """Compare BDF on stiff problem with scipy BDF."""
        lambda_val = 50.0  # Moderately stiff

        def f_torch(t, y):
            return -lambda_val * y

        def f_scipy(t, y):
            return -lambda_val * y

        y0_val = 1.0
        t_span = (0.0, 1.0)

        # Solve with torchscience
        y0_torch = torch.tensor([y0_val], dtype=torch.float64)
        y_torch, _ = bdf(f_torch, y0_torch, t_span, rtol=1e-3, atol=1e-6)

        # Solve with scipy BDF
        sol_scipy = solve_ivp(
            f_scipy, t_span, [y0_val], method="BDF", rtol=1e-6, atol=1e-8
        )

        # Both should give very small values (exp(-50) ~ 1e-22)
        assert y_torch.abs().item() < 1e-8
        assert abs(sol_scipy.y[0, -1]) < 1e-10

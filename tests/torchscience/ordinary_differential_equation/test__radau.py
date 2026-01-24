# tests/torchscience/integration/initial_value_problem/test__radau.py
"""Tests for Radau IIA (3-stage, order 5) ODE solver."""

import pytest
import torch
from tensordict import TensorDict

from torchscience.ordinary_differential_equation import (
    ConvergenceError,
    MaxStepsExceeded,
)
from torchscience.ordinary_differential_equation._radau import radau


class TestRadauBasic:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 => y(t) = exp(-t).

        This tests basic functionality of the Radau IIA solver.
        Order 5 method should achieve rtol=1e-5 accuracy.
        """

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = radau(
            decay,
            y0,
            t_span=(0.0, 1.0),
            rtol=1e-5,
            atol=1e-8,
        )

        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        # Order 5 method should be quite accurate
        assert torch.allclose(y_final, expected, rtol=1e-4)

    def test_harmonic_oscillator(self):
        """dx/dt = v, dv/dt = -x => x(t) = cos(t).

        Test 2D system: simple harmonic oscillator.
        After one period (2*pi), should return close to initial.
        """

        def oscillator(t, y):
            x, v = y[..., 0], y[..., 1]
            return torch.stack([v, -x], dim=-1)

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        t_period = 2 * 3.141592653589793  # 2*pi

        y_final, interp = radau(
            oscillator,
            y0,
            t_span=(0.0, t_period),
            rtol=1e-5,
            atol=1e-8,
        )

        # After one period, should return close to initial condition
        assert torch.allclose(y_final, y0, rtol=1e-3, atol=1e-4)

    def test_returns_interpolant(self):
        """Should return callable interpolant."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = radau(f, y0, t_span=(0.0, 1.0), rtol=1e-5, atol=1e-8)

        # Interpolant should be callable
        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_interpolant_endpoints(self):
        """Interpolant should match endpoints."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = radau(f, y0, t_span=(0.0, 1.0), rtol=1e-5, atol=1e-8)

        # At t=0, should match y0
        assert torch.allclose(interp(0.0), y0, atol=1e-6)
        # At t=1, should match y_final
        assert torch.allclose(interp(1.0), y_final, atol=1e-6)


class TestRadauStiff:
    def test_stiff_decay(self):
        """Radau should handle very stiff problems (lambda=10000).

        Radau IIA is L-stable and should handle extremely stiff problems
        much more efficiently than explicit methods.
        """
        lambda_val = 10000.0

        def stiff_decay(t, y):
            return -lambda_val * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = radau(
            stiff_decay,
            y0,
            t_span=(0.0, 0.001),  # 10 time constants
            rtol=1e-5,
            atol=1e-8,
        )

        # y_final should be approximately exp(-10000 * 0.001) = exp(-10) ~ 4.5e-5
        expected = torch.exp(torch.tensor([-10.0], dtype=torch.float64))
        # Should be very small
        assert y_final.abs().item() < 1e-3

    def test_robertson_problem(self):
        """Classic stiff benchmark - verify mass conservation.

        Robertson's chemical kinetics:
        dy1/dt = -k1*y1 + k2*y2*y3
        dy2/dt = k1*y1 - k2*y2*y3 - k3*y2^2
        dy3/dt = k3*y2^2

        with k1=0.04, k2=1e4, k3=3e7.
        """
        k1, k2, k3 = 0.04, 1e4, 3e7

        def robertson(t, y):
            y1, y2, y3 = y[0], y[1], y[2]
            dy1 = -k1 * y1 + k2 * y2 * y3
            dy2 = k1 * y1 - k2 * y2 * y3 - k3 * y2**2
            dy3 = k3 * y2**2
            return torch.stack([dy1, dy2, dy3])

        y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        y_final, _ = radau(
            robertson,
            y0,
            t_span=(0.0, 1.0),
            rtol=1e-5,
            atol=1e-8,
            max_steps=50000,
        )

        # Conservation: y1 + y2 + y3 = 1 (mass balance)
        total = y_final.sum()
        assert torch.allclose(
            total, torch.tensor(1.0, dtype=torch.float64), atol=1e-3
        )

        # All concentrations should be non-negative
        assert (y_final >= -1e-5).all()

    def test_robertson_long_integration(self):
        """Robertson problem integrated to t=1e5.

        Classic benchmark endpoint for Robertson problem.
        """
        k1, k2, k3 = 0.04, 1e4, 3e7

        def robertson(t, y):
            y1, y2, y3 = y[0], y[1], y[2]
            dy1 = -k1 * y1 + k2 * y2 * y3
            dy2 = k1 * y1 - k2 * y2 * y3 - k3 * y2**2
            dy3 = k3 * y2**2
            return torch.stack([dy1, dy2, dy3])

        y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        y_final, _ = radau(
            robertson,
            y0,
            t_span=(0.0, 1e5),
            rtol=1e-5,
            atol=1e-8,
            max_steps=100000,
        )

        # Conservation: y1 + y2 + y3 = 1 (mass balance)
        total = y_final.sum()
        assert torch.allclose(
            total, torch.tensor(1.0, dtype=torch.float64), atol=1e-2
        )

        # All concentrations should be non-negative
        assert (y_final >= -1e-4).all()

        # At t=1e5, y1 should be nearly depleted, y3 should dominate
        assert y_final[0].item() < 0.5  # y1 should decrease significantly
        assert y_final[2].item() > 0.5  # y3 should accumulate


class TestRadauAutograd:
    def test_gradient_through_solver(self):
        """Gradients should flow through Radau solver."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, _ = radau(f, y0, t_span=(0.0, 1.0), rtol=1e-5, atol=1e-8)

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
        _, interp = radau(f, y0, t_span=(0.0, 1.0), rtol=1e-5, atol=1e-8)

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

        y_final, _ = radau(f, y0, t_span=(0.0, 1.0), rtol=1e-5, atol=1e-8)

        loss = y_final.sum()
        loss.backward()

        assert y0.grad is not None
        # Gradient of exp(-t) wrt y0 is exp(-1)
        expected_grad = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        assert torch.allclose(y0.grad, expected_grad, rtol=0.05)


class TestRadauTensorDict:
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
        state_final, _ = radau(
            f, state0, t_span=(0.0, 1.0), rtol=1e-5, atol=1e-8
        )

        assert isinstance(state_final, TensorDict)
        assert "x" in state_final.keys()
        assert "v" in state_final.keys()

    def test_tensordict_interpolant(self):
        """Interpolant should work with TensorDict."""

        def f(t, state):
            return TensorDict({"x": -state["x"]})

        state0 = TensorDict({"x": torch.tensor([1.0], dtype=torch.float64)})
        _, interp = radau(f, state0, t_span=(0.0, 1.0), rtol=1e-5, atol=1e-8)

        state_mid = interp(0.5)
        assert isinstance(state_mid, TensorDict)
        assert "x" in state_mid.keys()


class TestRadauErrorHandling:
    def test_max_steps_exceeded_throws(self):
        """Should raise MaxStepsExceeded when exceeding step limit."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(MaxStepsExceeded):
            radau(
                f, y0, t_span=(0.0, 100.0), max_steps=5, rtol=1e-5, atol=1e-8
            )

    def test_max_steps_exceeded_no_throw(self):
        """Should return NaN when exceeding step limit with throw=False."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, interp = radau(
            f,
            y0,
            t_span=(0.0, 100.0),
            max_steps=5,
            throw=False,
            rtol=1e-5,
            atol=1e-8,
        )

        assert torch.isnan(y_final).all()
        assert interp.success is not None
        assert not interp.success.all()

    def test_convergence_error_thrown(self):
        """Should raise ConvergenceError or MaxStepsExceeded when solver fails."""

        def difficult(t, y):
            # Extremely stiff dynamics with very tight tolerances
            return -10000 * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Either ConvergenceError (Newton fails) or MaxStepsExceeded (too many steps)
        # are valid failure modes for this difficult problem
        with pytest.raises((ConvergenceError, MaxStepsExceeded)):
            radau(
                difficult,
                y0,
                t_span=(0.0, 1.0),
                dt_min=1e-10,
                rtol=1e-12,
                atol=1e-14,
                max_newton_iter=2,
            )


class TestRadauBackwardIntegration:
    def test_backward_exponential(self):
        """Integrate backwards: y(1) = e^-1 => y(0) = 1."""

        def f(t, y):
            return -y

        y1 = torch.tensor([torch.exp(torch.tensor(-1.0))], dtype=torch.float64)
        y0_recovered, interp = radau(
            f, y1, t_span=(1.0, 0.0), rtol=1e-4, atol=1e-6
        )

        expected = torch.tensor([1.0], dtype=torch.float64)
        assert torch.allclose(y0_recovered, expected, rtol=0.05)


class TestRadauStepSizeControl:
    def test_dt0_is_used(self):
        """Initial step size hint should be used."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # With small dt0, should still complete
        y_final, _ = radau(
            f, y0, t_span=(0.0, 1.0), dt0=0.001, rtol=1e-5, atol=1e-8
        )
        assert not torch.isnan(y_final).any()

    def test_dt_max_limits_step(self):
        """dt_max should limit maximum step size."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, interp = radau(
            f, y0, t_span=(0.0, 1.0), dt_max=0.01, rtol=1e-5, atol=1e-8
        )

        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        assert torch.allclose(y_final, expected, rtol=1e-4)


class TestRadauVanDerPol:
    """Van der Pol oscillator - stiffness varies with mu.

    Radau IIA should handle highly stiff Van der Pol better than BDF.
    """

    def test_van_der_pol_mu_10(self):
        """Test Van der Pol with mu=10 (moderately stiff)."""
        mu = 10.0

        def van_der_pol(t, y):
            y1, y2 = y[0], y[1]
            dy1 = y2
            dy2 = mu * (1 - y1**2) * y2 - y1
            return torch.stack([dy1, dy2])

        y0 = torch.tensor([2.0, 0.0], dtype=torch.float64)
        y_final, _ = radau(
            van_der_pol,
            y0,
            t_span=(0.0, 5.0),
            rtol=1e-5,
            atol=1e-8,
            max_steps=50000,
        )

        assert not torch.isnan(y_final).any()
        assert y_final[0].abs().item() < 5  # Position bounded

    def test_van_der_pol_mu_100(self):
        """Test Van der Pol with mu=100 (classic stiff benchmark).

        Radau IIA should handle this better than BDF.
        """
        mu = 100.0

        def van_der_pol(t, y):
            y1, y2 = y[0], y[1]
            dy1 = y2
            dy2 = mu * (1 - y1**2) * y2 - y1
            return torch.stack([dy1, dy2])

        y0 = torch.tensor([2.0, 0.0], dtype=torch.float64)
        y_final, _ = radau(
            van_der_pol,
            y0,
            t_span=(0.0, 2.0),
            rtol=1e-5,
            atol=1e-8,
            max_steps=100000,
        )

        assert not torch.isnan(y_final).any()
        assert y_final[0].abs().item() < 5  # Position bounded


scipy = pytest.importorskip("scipy")
from scipy.integrate import solve_ivp


class TestRadauSciPyComparison:
    def test_exponential_decay_matches_scipy(self):
        """Compare Radau results with scipy Radau."""

        def f_torch(t, y):
            return -y

        def f_scipy(t, y):
            return -y

        y0_val = 1.0
        t_span = (0.0, 5.0)

        # Solve with torchscience
        y0_torch = torch.tensor([y0_val], dtype=torch.float64)
        y_torch, _ = radau(f_torch, y0_torch, t_span, rtol=1e-5, atol=1e-8)

        # Solve with scipy Radau
        sol_scipy = solve_ivp(
            f_scipy, t_span, [y0_val], method="Radau", rtol=1e-8, atol=1e-10
        )

        # Both should give similar results
        assert torch.allclose(
            y_torch,
            torch.tensor(sol_scipy.y[:, -1], dtype=torch.float64),
            rtol=1e-3,
        )

    def test_stiff_problem_matches_scipy(self):
        """Compare Radau on stiff problem with scipy Radau."""
        lambda_val = 1000.0

        def f_torch(t, y):
            return -lambda_val * y

        def f_scipy(t, y):
            return -lambda_val * y

        y0_val = 1.0
        t_span = (0.0, 0.01)

        # Solve with torchscience
        y0_torch = torch.tensor([y0_val], dtype=torch.float64)
        y_torch, _ = radau(f_torch, y0_torch, t_span, rtol=1e-5, atol=1e-8)

        # Solve with scipy Radau
        sol_scipy = solve_ivp(
            f_scipy, t_span, [y0_val], method="Radau", rtol=1e-8, atol=1e-10
        )

        # Both should give very small values
        assert torch.allclose(
            y_torch,
            torch.tensor(sol_scipy.y[:, -1], dtype=torch.float64),
            rtol=0.1,
        )

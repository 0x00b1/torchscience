"""Tests for Implicit Midpoint symplectic integrator."""

import math

import pytest
import torch

from torchscience.ordinary_differential_equation import ConvergenceError
from torchscience.ordinary_differential_equation._implicit_midpoint import (
    implicit_midpoint,
)


class TestImplicitMidpointBasic:
    def test_harmonic_oscillator(self):
        """Implicit midpoint on harmonic oscillator (general ODE form)."""

        def f(t, y):
            # y = [q, p], f = [p, -q]
            q, p = y[0], y[1]
            return torch.stack([p, -q])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        y_final, interp = implicit_midpoint(
            f, y0, t_span=(0.0, 2 * math.pi), dt=0.1
        )

        # After one period, should return close to initial
        assert torch.allclose(y_final, y0, atol=0.05)

    def test_energy_conservation(self):
        """Implicit midpoint preserves quadratic invariants exactly."""

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -q])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        def energy(y):
            return 0.5 * y[0] ** 2 + 0.5 * y[1] ** 2

        E0 = energy(y0).item()

        # Long integration
        y_final, _ = implicit_midpoint(f, y0, t_span=(0.0, 100.0), dt=0.1)

        E_final = energy(y_final).item()

        # Implicit midpoint preserves quadratic invariants up to Newton tolerance
        assert abs(E_final - E0) < 1e-6

    def test_stiff_hamiltonian(self):
        """Implicit midpoint handles stiff Hamiltonian systems (A-stable)."""
        # Stiff harmonic oscillator: H = p^2/2 + omega^2*q^2/2
        # omega = sqrt(1000) ~ 31.6, so momentum can reach ~31.6 * q_max
        omega_sq = 1000.0

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -omega_sq * q])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        # Stiff system energy: E = p^2/2 + omega^2*q^2/2
        def energy(y):
            return 0.5 * y[1] ** 2 + 0.5 * omega_sq * y[0] ** 2

        E0 = energy(y0).item()

        # Even with large step relative to stiffness, should be stable
        y_final, _ = implicit_midpoint(
            f, y0, t_span=(0.0, 1.0), dt=0.1, newton_tol=1e-10
        )

        E_final = energy(y_final).item()

        # Should remain bounded and stable (explicit methods would explode)
        # Implicit midpoint preserves quadratic invariants (energy)
        assert not torch.isnan(y_final).any()
        assert (
            abs(E_final - E0) / E0 < 1e-6
        )  # Energy conserved to Newton tolerance

    def test_exponential_decay(self):
        """Test on standard ODE: dy/dt = -y."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = implicit_midpoint(f, y0, t_span=(0.0, 1.0), dt=0.1)

        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        # Implicit midpoint is 2nd order, should be reasonably accurate
        assert torch.allclose(y_final, expected, rtol=0.05)

    def test_returns_interpolant(self):
        """Interpolant should return states at queried times."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = implicit_midpoint(f, y0, t_span=(0.0, 1.0), dt=0.1)

        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape
        # Interpolant at midpoint should be between y0 and y_final
        assert y_mid[0] < y0[0]
        assert y_mid[0] > y_final[0]

    def test_interpolant_endpoints(self):
        """Interpolant should match at endpoints."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        y_final, interp = implicit_midpoint(f, y0, t_span=(0.0, 1.0), dt=0.1)

        assert torch.allclose(interp(0.0), y0, atol=1e-10)
        assert torch.allclose(interp(1.0), y_final, atol=1e-10)


class TestImplicitMidpointAutograd:
    def test_gradient_through_solver(self):
        """Gradients should flow through implicit midpoint solver."""
        # Use a scalar parameter (0-dim tensor) to avoid shape mismatch in torch.stack
        k = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -k * q])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        y_final, _ = implicit_midpoint(f, y0, t_span=(0.0, 1.0), dt=0.1)

        loss = y_final.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()

    def test_gradient_through_interpolant(self):
        """Gradients should flow through interpolant."""
        theta = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        _, interp = implicit_midpoint(f, y0, t_span=(0.0, 1.0), dt=0.1)

        y_mid = interp(0.5)
        loss = y_mid.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad).any()


class TestImplicitMidpointNewton:
    def test_convergence_error_thrown(self):
        """Should raise ConvergenceError when Newton fails."""

        # Difficult dynamics that make Newton hard to converge
        def f(t, y):
            return y**3 - 100 * y

        y0 = torch.tensor([0.1], dtype=torch.float64)

        with pytest.raises(ConvergenceError):
            implicit_midpoint(
                f,
                y0,
                t_span=(0.0, 1.0),
                dt=0.5,
                newton_tol=1e-15,
                max_newton_iter=2,
            )

    def test_convergence_failure_no_throw(self):
        """Should return NaN when Newton fails with throw=False."""

        def f(t, y):
            return y**3 - 100 * y

        y0 = torch.tensor([0.1], dtype=torch.float64)

        y_final, interp = implicit_midpoint(
            f,
            y0,
            t_span=(0.0, 1.0),
            dt=0.5,
            newton_tol=1e-15,
            max_newton_iter=2,
            throw=False,
        )

        # Either NaN or success flag is False
        assert torch.isnan(y_final).any() or interp.success == False


class TestImplicitMidpointBackwardIntegration:
    def test_backward_integration(self):
        """Should support backward integration (t1 < t0)."""

        def f(t, y):
            return -y

        # Start from y(1) = exp(-1) and integrate backwards to t=0
        y1 = torch.tensor([torch.exp(torch.tensor(-1.0))], dtype=torch.float64)
        y0_recovered, _ = implicit_midpoint(f, y1, t_span=(1.0, 0.0), dt=0.1)

        expected = torch.tensor([1.0], dtype=torch.float64)
        # Should recover approximately y(0) = 1
        assert torch.allclose(y0_recovered, expected, rtol=0.1)


class TestImplicitMidpointMaxSteps:
    def test_max_steps_exceeded(self):
        """Should raise MaxStepsExceeded when exceeding step limit."""
        from torchscience.ordinary_differential_equation import (  # noqa: PLC0415
            MaxStepsExceeded,
        )

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(MaxStepsExceeded):
            implicit_midpoint(
                f,
                y0,
                t_span=(0.0, 100.0),
                dt=0.01,
                max_steps=10,  # Will need more than 10 steps
            )

    def test_max_steps_no_throw(self):
        """Should return NaN when exceeding step limit with throw=False."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, interp = implicit_midpoint(
            f, y0, t_span=(0.0, 100.0), dt=0.01, max_steps=10, throw=False
        )

        assert torch.isnan(y_final).any() or interp.success == False

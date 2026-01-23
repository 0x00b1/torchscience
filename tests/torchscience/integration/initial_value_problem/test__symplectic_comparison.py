"""Compare symplectic integrators on benchmark problems."""

import math

import pytest
import torch

from torchscience.integration.initial_value_problem import (
    implicit_midpoint,
    stormer_verlet,
    yoshida4,
)


class TestSymplecticComparison:
    """Compare all symplectic integrators on harmonic oscillator."""

    def test_all_preserve_energy(self):
        """All symplectic integrators should preserve energy."""

        def grad_V(t, q):
            return q

        def grad_T(t, p):
            return p

        def f(t, y):
            return torch.stack([y[1], -y[0]])

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)
        y0 = torch.stack([q0, p0]).squeeze()

        def energy(q, p):
            return 0.5 * p**2 + 0.5 * q**2

        def energy_y(y):
            return 0.5 * y[0] ** 2 + 0.5 * y[1] ** 2

        E0 = energy(q0, p0).item()

        # Long integration: 50 periods
        T = 50 * 2 * math.pi
        dt = 0.1

        # Stormer-Verlet
        q_sv, p_sv, _ = stormer_verlet(grad_V, grad_T, q0, p0, (0, T), dt)
        E_sv = energy(q_sv, p_sv).item()

        # Yoshida 4
        q_y4, p_y4, _ = yoshida4(grad_V, grad_T, q0, p0, (0, T), dt)
        E_y4 = energy(q_y4, p_y4).item()

        # Implicit Midpoint
        y_im, _ = implicit_midpoint(f, y0, (0, T), dt)
        E_im = energy_y(y_im).item()

        # All should preserve energy (bounded oscillation, not drift)
        assert abs(E_sv - E0) < 0.5, f"Verlet energy drift: {abs(E_sv - E0)}"
        assert abs(E_y4 - E0) < 0.1, f"Yoshida4 energy drift: {abs(E_y4 - E0)}"
        assert abs(E_im - E0) < 1e-5, (
            f"Implicit midpoint energy drift: {abs(E_im - E0)}"
        )

    def test_yoshida4_more_accurate_than_verlet(self):
        """Yoshida4 should be more accurate than Verlet at same step size."""

        def grad_V(t, q):
            return q

        def grad_T(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)

        # One period
        T = 2 * math.pi
        dt = 0.2  # Relatively large step

        q_sv, p_sv, _ = stormer_verlet(grad_V, grad_T, q0, p0, (0, T), dt)
        q_y4, p_y4, _ = yoshida4(grad_V, grad_T, q0, p0, (0, T), dt)

        # Error after one period (should return to initial)
        err_sv = torch.sqrt((q_sv - q0) ** 2 + (p_sv - p0) ** 2).item()
        err_y4 = torch.sqrt((q_y4 - q0) ** 2 + (p_y4 - p0) ** 2).item()

        # Yoshida4 (4th order) should be more accurate than Verlet (2nd order)
        assert err_y4 < err_sv, (
            f"Yoshida4 error {err_y4} should be < Verlet error {err_sv}"
        )

    def test_implicit_midpoint_handles_stiff(self):
        """Implicit midpoint should handle stiff problems that explicit can't."""
        # Stiff oscillator: omega = 10
        omega = 10.0

        def f(t, y):
            return torch.stack([y[1], -(omega**2) * y[0]])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        # Step size that would make explicit methods unstable
        # For explicit Euler, stability limit is 2/omega = 0.2
        # We use dt = 0.1, which is marginally stable for explicit but
        # implicit midpoint handles it robustly
        dt = 0.1

        y_final, _ = implicit_midpoint(f, y0, (0, 1.0), dt, newton_tol=1e-10)

        # Energy should be conserved: E = 0.5 * p^2 + 0.5 * omega^2 * q^2
        # Initial energy: E0 = 0.5 * omega^2 * 1^2 = 50
        E0 = 0.5 * omega**2 * y0[0] ** 2 + 0.5 * y0[1] ** 2
        E_final = 0.5 * omega**2 * y_final[0] ** 2 + 0.5 * y_final[1] ** 2

        # Should remain bounded (position amplitude should stay ~1)
        assert y_final[0].abs() < 1.5, (
            f"Position {y_final[0].item()} should stay bounded"
        )
        # Momentum amplitude should be at most omega * position_amplitude
        assert y_final[1].abs() < omega * 1.5, (
            f"Momentum {y_final[1].item()} should stay bounded"
        )
        assert not torch.isnan(y_final).any()
        # Energy should be approximately conserved
        assert abs(E_final.item() - E0.item()) < 1.0, (
            f"Energy drift: {abs(E_final.item() - E0.item())}"
        )


class TestKeplerProblem:
    """Test all symplectic integrators on Kepler (two-body) problem."""

    @pytest.fixture
    def kepler_setup(self):
        """Setup for Kepler problem."""

        def grad_V(t, q):
            r = torch.norm(q)
            return q / (r**3 + 1e-10)

        def grad_T(t, p):
            return p

        def f(t, y):
            q = y[:2]
            p = y[2:]
            r = torch.norm(q)
            return torch.cat([p, -q / (r**3 + 1e-10)])

        # Circular orbit
        q0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        p0 = torch.tensor([0.0, 1.0], dtype=torch.float64)
        y0 = torch.cat([q0, p0])

        return grad_V, grad_T, f, q0, p0, y0

    def test_circular_orbit_preserved(self, kepler_setup):
        """Circular orbit should remain circular for all integrators."""
        grad_V, grad_T, f, q0, p0, y0 = kepler_setup

        T = 2 * math.pi  # One orbit
        dt = 0.02

        # Stormer-Verlet
        q_sv, p_sv, _ = stormer_verlet(grad_V, grad_T, q0, p0, (0, T), dt)

        # Yoshida 4
        q_y4, p_y4, _ = yoshida4(grad_V, grad_T, q0, p0, (0, T), dt)

        # Implicit Midpoint
        y_im, _ = implicit_midpoint(f, y0, (0, T), dt)
        q_im, p_im = y_im[:2], y_im[2:]

        # All should return close to initial
        assert torch.allclose(q_sv, q0, atol=0.1)
        assert torch.allclose(q_y4, q0, atol=0.05)  # More accurate
        assert torch.allclose(q_im, q0, atol=0.1)

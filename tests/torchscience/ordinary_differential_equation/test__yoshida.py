"""Tests for Yoshida 4th-order symplectic integrator."""

import math

import torch

from torchscience.ordinary_differential_equation._yoshida import yoshida4


class TestYoshida4Basic:
    def test_harmonic_oscillator(self):
        """Yoshida4 should be more accurate than Verlet for harmonic oscillator."""

        def grad_potential(t, q):
            return q

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)

        # With same step size, Yoshida4 should be more accurate
        q_final, p_final, _ = yoshida4(
            grad_potential,
            grad_kinetic,
            q0,
            p0,
            t_span=(0.0, 2 * math.pi),
            dt=0.1,  # Larger step than Verlet test
        )

        # Should still be accurate after one period
        assert torch.allclose(q_final, q0, atol=0.01)
        assert torch.allclose(p_final, p0, atol=0.01)

    def test_energy_conservation_better_than_verlet(self):
        """Yoshida4 should have smaller energy oscillations than Verlet."""

        def grad_potential(t, q):
            return q

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)

        def energy(q, p):
            return 0.5 * p**2 + 0.5 * q**2

        E0 = energy(q0, p0).item()

        # Integrate for 10 periods with relatively large step
        q_final, p_final, _ = yoshida4(
            grad_potential,
            grad_kinetic,
            q0,
            p0,
            t_span=(0.0, 20 * math.pi),
            dt=0.1,
        )

        E_final = energy(q_final, p_final).item()

        # 4th order method should have O(h^4) energy error
        assert abs(E_final - E0) < 0.01

    def test_kepler_problem(self):
        """Kepler problem with 4th-order accuracy."""

        def grad_potential(t, q):
            r = torch.norm(q)
            return q / (r**3 + 1e-10)

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        p0 = torch.tensor([0.0, 1.0], dtype=torch.float64)

        q_final, p_final, _ = yoshida4(
            grad_potential,
            grad_kinetic,
            q0,
            p0,
            t_span=(0.0, 2 * math.pi),
            dt=0.05,
        )

        assert torch.allclose(q_final, q0, atol=0.05)
        assert torch.allclose(p_final, p0, atol=0.05)

    def test_backward_integration(self):
        """Test backward time integration."""

        def grad_potential(t, q):
            return q

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)

        # Forward integration
        q_fwd, p_fwd, _ = yoshida4(
            grad_potential, grad_kinetic, q0, p0, t_span=(0.0, 1.0), dt=0.01
        )

        # Backward integration from forward result
        q_back, p_back, _ = yoshida4(
            grad_potential,
            grad_kinetic,
            q_fwd,
            p_fwd,
            t_span=(1.0, 0.0),
            dt=0.01,
        )

        # Should return to initial state (symplectic integrators are time-reversible)
        assert torch.allclose(q_back, q0, atol=1e-5)
        assert torch.allclose(p_back, p0, atol=1e-5)


class TestYoshida4Interpolant:
    def test_returns_interpolant(self):
        """Should return callable interpolant for (q, p)."""

        def grad_potential(t, q):
            return q

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)

        q_final, p_final, interp = yoshida4(
            grad_potential, grad_kinetic, q0, p0, t_span=(0.0, 1.0), dt=0.1
        )

        q_mid, p_mid = interp(0.5)
        assert q_mid.shape == q0.shape
        assert p_mid.shape == p0.shape

    def test_interpolant_endpoints(self):
        """Interpolant should match initial and final states at endpoints."""

        def grad_potential(t, q):
            return q

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)

        q_final, p_final, interp = yoshida4(
            grad_potential, grad_kinetic, q0, p0, t_span=(0.0, 1.0), dt=0.1
        )

        q_start, p_start = interp(0.0)
        q_end, p_end = interp(1.0)

        assert torch.allclose(q_start, q0, atol=1e-10)
        assert torch.allclose(p_start, p0, atol=1e-10)
        assert torch.allclose(q_end, q_final, atol=1e-10)
        assert torch.allclose(p_end, p_final, atol=1e-10)


class TestYoshida4Autograd:
    def test_gradient_through_solver(self):
        """Gradients should flow through Yoshida4 solver."""
        k = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def grad_potential(t, q):
            return k * q  # V = k*q^2/2

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)

        q_final, p_final, _ = yoshida4(
            grad_potential, grad_kinetic, q0, p0, t_span=(0.0, 1.0), dt=0.1
        )

        loss = q_final.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()

    def test_gradient_wrt_initial_conditions(self):
        """Gradients should flow through initial conditions."""

        def grad_potential(t, q):
            return q

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        p0 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

        q_final, p_final, _ = yoshida4(
            grad_potential, grad_kinetic, q0, p0, t_span=(0.0, 1.0), dt=0.1
        )

        loss = q_final.sum() + p_final.sum()
        loss.backward()

        assert q0.grad is not None
        assert p0.grad is not None
        assert not torch.isnan(q0.grad).any()
        assert not torch.isnan(p0.grad).any()


class TestYoshida4Multidimensional:
    def test_2d_harmonic_oscillator(self):
        """Test with 2D position and momentum."""

        def grad_potential(t, q):
            return q  # Isotropic harmonic oscillator

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        p0 = torch.tensor([0.0, 1.0], dtype=torch.float64)

        q_final, p_final, _ = yoshida4(
            grad_potential,
            grad_kinetic,
            q0,
            p0,
            t_span=(0.0, 2 * math.pi),
            dt=0.05,
        )

        # After one period, should return close to initial
        assert torch.allclose(q_final, q0, atol=0.01)
        assert torch.allclose(p_final, p0, atol=0.01)

    def test_anisotropic_oscillator(self):
        """Test with different frequencies in different dimensions."""

        # Anisotropic: omega_x = 1, omega_y = 2
        def grad_potential(t, q):
            return torch.stack([q[0], 4.0 * q[1]])

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0, 0.5], dtype=torch.float64)
        p0 = torch.tensor([0.0, 0.0], dtype=torch.float64)

        def energy(q, p):
            return 0.5 * (p[0] ** 2 + p[1] ** 2) + 0.5 * (
                q[0] ** 2 + 4 * q[1] ** 2
            )

        E0 = energy(q0, p0).item()

        # Integrate for several periods
        q_final, p_final, _ = yoshida4(
            grad_potential,
            grad_kinetic,
            q0,
            p0,
            t_span=(0.0, 10 * math.pi),
            dt=0.05,
        )

        E_final = energy(q_final, p_final).item()

        # Energy should be conserved
        assert abs(E_final - E0) < 0.01


class TestYoshida4SpecialCases:
    def test_zero_initial_momentum(self):
        """Test with zero initial momentum (starts at rest)."""

        def grad_potential(t, q):
            return q

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)

        q_final, p_final, _ = yoshida4(
            grad_potential, grad_kinetic, q0, p0, t_span=(0.0, 0.5), dt=0.01
        )

        # Should have valid output
        assert not torch.isnan(q_final).any()
        assert not torch.isnan(p_final).any()

    def test_large_step_size(self):
        """Test stability with larger step sizes."""

        def grad_potential(t, q):
            return q

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)

        # Larger step size (less accurate but should be stable)
        q_final, p_final, _ = yoshida4(
            grad_potential,
            grad_kinetic,
            q0,
            p0,
            t_span=(0.0, 2 * math.pi),
            dt=0.2,
        )

        # Should still return to near initial (symplectic property)
        # 4th order method should be better than Verlet even with large step
        assert torch.allclose(q_final, q0, atol=0.1)
        assert torch.allclose(p_final, p0, atol=0.1)

    def test_throw_false_success_attribute(self):
        """Test that throw=False sets success attribute on interpolant."""

        def grad_potential(t, q):
            return q

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)

        q_final, p_final, interp = yoshida4(
            grad_potential,
            grad_kinetic,
            q0,
            p0,
            t_span=(0.0, 1.0),
            dt=0.1,
            throw=False,
        )

        assert hasattr(interp, "success")
        assert interp.success is not None
        assert interp.success.item() is True


class TestYoshida4OrderVerification:
    def test_fourth_order_convergence(self):
        """Verify 4th-order convergence by halving step size."""

        def grad_potential(t, q):
            return q

        def grad_kinetic(t, p):
            return p

        q0 = torch.tensor([1.0], dtype=torch.float64)
        p0 = torch.tensor([0.0], dtype=torch.float64)

        # Exact solution at t=1: q = cos(1), p = -sin(1)
        q_exact = torch.tensor([math.cos(1.0)], dtype=torch.float64)

        # Test with two step sizes
        dt1 = 0.1
        dt2 = 0.05  # Half the step size

        q1, _, _ = yoshida4(
            grad_potential, grad_kinetic, q0, p0, t_span=(0.0, 1.0), dt=dt1
        )
        q2, _, _ = yoshida4(
            grad_potential, grad_kinetic, q0, p0, t_span=(0.0, 1.0), dt=dt2
        )

        error1 = (q1 - q_exact).abs().item()
        error2 = (q2 - q_exact).abs().item()

        # For 4th order method, halving step size should reduce error by ~16x
        # Allow some tolerance for implementation details
        ratio = error1 / (error2 + 1e-15)
        assert ratio > 10, f"Expected ~16x error reduction, got {ratio:.1f}x"

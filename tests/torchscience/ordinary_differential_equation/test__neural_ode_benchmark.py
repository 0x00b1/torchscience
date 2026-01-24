"""Benchmark tests comparing neural ODE solvers."""

import math

import torch

from torchscience.ordinary_differential_equation import (
    adjoint,
    asynchronous_leapfrog,
    euler,
    reversible_heun,
    runge_kutta_4,
)


class TestNeuralODESpiral:
    """Test on classic neural ODE spiral problem."""

    def test_all_solvers_converge(self):
        """All neural ODE solvers should produce similar results."""
        # Simple rotation dynamics
        W = torch.tensor(
            [[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=False
        )

        def f(t, y):
            return y @ W.T

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        results = {}

        # Standard solvers
        results["euler"], _ = euler(f, y0, t_span=(0, 1), dt=0.01)
        results["rk4"], _ = runge_kutta_4(f, y0, t_span=(0, 1), dt=0.1)
        results["rev_heun"], _ = reversible_heun(f, y0, t_span=(0, 1), dt=0.05)
        results["ala"], _ = asynchronous_leapfrog(
            f, y0, t_span=(0, 1), dt=0.05
        )

        # All should be close to RK4 (most accurate)
        ref = results["rk4"]
        for name, y in results.items():
            if name != "rk4":
                assert torch.allclose(y, ref, atol=0.1), (
                    f"{name} diverged from RK4"
                )

    def test_gradient_accuracy_comparison(self):
        """Compare gradient accuracy across solvers."""
        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        grads = {}

        # Reversible Heun
        W_rev = torch.tensor(
            [[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )

        def f_rev(t, y):
            return y @ W_rev.T

        y_rev, _ = reversible_heun(f_rev, y0, t_span=(0, 1), dt=0.05)
        y_rev.sum().backward()
        grads["rev_heun"] = W_rev.grad.clone()

        # ALA
        W_ala = torch.tensor(
            [[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )

        def f_ala(t, y):
            return y @ W_ala.T

        y_ala, _ = asynchronous_leapfrog(f_ala, y0, t_span=(0, 1), dt=0.05)
        y_ala.sum().backward()
        grads["ala"] = W_ala.grad.clone()

        # Standard RK4 with autograd
        W_rk4 = torch.tensor(
            [[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )

        def f_rk4(t, y):
            return y @ W_rk4.T

        y_rk4, _ = runge_kutta_4(f_rk4, y0, t_span=(0, 1), dt=0.1)
        y_rk4.sum().backward()
        grads["rk4"] = W_rk4.grad.clone()

        # Gradients should be similar
        for name, grad in grads.items():
            if name != "rk4":
                assert torch.allclose(grad, grads["rk4"], atol=0.2), (
                    f"{name} gradient differs from RK4"
                )


class TestMemoryEfficiency:
    """Test memory efficiency of reversible solvers."""

    def test_reversible_heun_long_integration(self):
        """Reversible Heun should handle long integrations."""
        k = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Long integration: 1000 steps
        y_final, _ = reversible_heun(f, y0, t_span=(0, 100), dt=0.1)
        loss = y_final.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()

    def test_ala_long_integration(self):
        """ALA should handle long integrations."""
        k = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)

        def f(t, y):
            return -k * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Long integration
        y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0, 100), dt=0.1)
        loss = y_final.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()


class TestAdjointComparison:
    """Compare reversible solvers with adjoint method."""

    def test_adjoint_vs_reversible_heun(self):
        """Adjoint and reversible_heun should give similar gradients."""
        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Reversible Heun gradient
        k_rev = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f_rev(t, y):
            return -k_rev * y

        y_rev, _ = reversible_heun(f_rev, y0, t_span=(0, 1), dt=0.05)
        y_rev.sum().backward()
        grad_rev = k_rev.grad.item()

        # Adjoint RK4 gradient
        k_adj = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f_adj(t, y):
            return -k_adj * y

        adjoint_rk4 = adjoint(runge_kutta_4)
        y_adj, _ = adjoint_rk4(f_adj, y0, t_span=(0, 1), dt=0.05)
        y_adj.sum().backward()
        grad_adj = k_adj.grad.item()

        # Should be similar (within numerical tolerance)
        assert abs(grad_rev - grad_adj) < 0.1


class TestConvergenceOrders:
    """Verify convergence orders of different solvers."""

    def test_euler_first_order(self):
        """Euler should have 1st order convergence."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        errors = []
        dts = [0.1, 0.05, 0.025]

        for dt in dts:
            y_final, _ = euler(f, y0, t_span=(0, 1), dt=dt)
            error = (y_final - exact).abs().item()
            errors.append(error)

        # Check convergence order: ratio ~2 for 1st order
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        assert 1.5 < ratio1 < 2.5
        assert 1.5 < ratio2 < 2.5

    def test_reversible_heun_second_order(self):
        """Reversible Heun should have 2nd order convergence."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        errors = []
        dts = [0.1, 0.05, 0.025]

        for dt in dts:
            y_final, _ = reversible_heun(f, y0, t_span=(0, 1), dt=dt)
            error = (y_final - exact).abs().item()
            errors.append(error)

        # Check convergence order: ratio ~4 for 2nd order
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        assert 2.5 < ratio1 < 6.0
        assert 2.5 < ratio2 < 6.0

    def test_ala_second_order(self):
        """ALA should have 2nd order convergence."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        errors = []
        dts = [0.1, 0.05, 0.025]

        for dt in dts:
            y_final, _ = asynchronous_leapfrog(f, y0, t_span=(0, 1), dt=dt)
            error = (y_final - exact).abs().item()
            errors.append(error)

        # Check convergence order: ratio ~4 for 2nd order
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        assert 2.5 < ratio1 < 6.0
        assert 2.5 < ratio2 < 6.0


class TestHarmonicOscillator:
    """Test solvers on harmonic oscillator (energy conservation)."""

    def test_solvers_complete_period(self):
        """All solvers should approximately complete one period."""

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -q])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        t_span = (0.0, 2 * math.pi)

        # Test each solver
        y_euler, _ = euler(f, y0, t_span=t_span, dt=0.01)
        y_rk4, _ = runge_kutta_4(f, y0, t_span=t_span, dt=0.1)
        y_rev, _ = reversible_heun(f, y0, t_span=t_span, dt=0.05)
        y_ala, _ = asynchronous_leapfrog(f, y0, t_span=t_span, dt=0.05)

        # RK4 should be most accurate
        assert torch.allclose(y_rk4, y0, atol=0.01)

        # Others should be reasonable
        assert torch.allclose(y_rev, y0, atol=0.1)
        assert torch.allclose(y_ala, y0, atol=0.1)

    def test_energy_drift(self):
        """Check energy drift over multiple periods."""

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -q])

        def energy(y):
            q, p = y[0], y[1]
            return 0.5 * (q**2 + p**2)

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        E0 = energy(y0).item()

        # Integrate for 10 periods
        t_span = (0.0, 20 * math.pi)

        y_rev, _ = reversible_heun(f, y0, t_span=t_span, dt=0.05)
        y_ala, _ = asynchronous_leapfrog(f, y0, t_span=t_span, dt=0.05)

        E_rev = energy(y_rev).item()
        E_ala = energy(y_ala).item()

        # Energy should not drift too much
        assert abs(E_rev - E0) / E0 < 0.1  # <10% drift
        assert abs(E_ala - E0) / E0 < 0.1  # <10% drift

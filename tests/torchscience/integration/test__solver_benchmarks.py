"""Comprehensive benchmark tests for all ODE solvers."""

import math

import torch

from torchscience.integration import (
    adams,
    bdf,
    dop853,
    dormand_prince_5,
    reversible_heun,
    runge_kutta_4,
    stormer_verlet,
    yoshida4,
)


class TestVanDerPolOscillator:
    """Van der Pol oscillator benchmarks (stiff for large mu)."""

    def test_nonstiff_explicit_solvers(self):
        """Non-stiff Van der Pol (mu=1) should work with explicit solvers."""
        mu = 1.0

        def f(t, y):
            x, v = y[0], y[1]
            return torch.stack([v, mu * (1 - x**2) * v - x])

        y0 = torch.tensor([2.0, 0.0], dtype=torch.float64)

        # All explicit solvers should handle this
        results = {}
        results["rk4"], _ = runge_kutta_4(f, y0, t_span=(0, 10), dt=0.01)
        results["dp5"], _ = dormand_prince_5(f, y0, t_span=(0, 10))
        results["dop853"], _ = dop853(f, y0, t_span=(0, 10))
        # Adams needs looser tolerances for oscillatory problems
        results["adams"], _ = adams(
            f, y0, t_span=(0, 10), rtol=1e-4, atol=1e-6
        )

        # All should complete without NaN
        for name, y in results.items():
            assert not torch.isnan(y).any(), f"{name} produced NaN"

    def test_stiff_requires_implicit(self):
        """Stiff Van der Pol (mu=100) requires implicit solvers."""
        mu = 100.0

        def f(t, y):
            x, v = y[0], y[1]
            return torch.stack([v, mu * (1 - x**2) * v - x])

        y0 = torch.tensor([2.0, 0.0], dtype=torch.float64)

        # Implicit solvers should handle this
        y_bdf, _ = bdf(f, y0, t_span=(0, 2 * mu), rtol=1e-4, atol=1e-6)

        assert not torch.isnan(y_bdf).any()


class TestKeplerOrbit:
    """Kepler two-body problem (energy conservation)."""

    def test_symplectic_energy_conservation(self):
        """Symplectic integrators should conserve energy."""

        def grad_V(t, q):
            r = torch.norm(q)
            return q / (r**3)  # -grad(-1/r) = q/r^3

        def grad_T(t, p):
            return p  # grad(p^2/2) = p

        # Circular orbit initial conditions
        q0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        p0 = torch.tensor([0.0, 1.0], dtype=torch.float64)

        def energy(q, p):
            return 0.5 * (p**2).sum() - 1.0 / torch.norm(q)

        E0 = energy(q0, p0).item()

        # Integrate for several orbits
        q_sv, p_sv, _ = stormer_verlet(
            grad_V, grad_T, q0, p0, t_span=(0, 20 * math.pi), dt=0.01
        )
        q_y4, p_y4, _ = yoshida4(
            grad_V, grad_T, q0, p0, t_span=(0, 20 * math.pi), dt=0.05
        )

        E_sv = energy(q_sv, p_sv).item()
        E_y4 = energy(q_y4, p_y4).item()

        # Energy drift should be bounded
        assert abs(E_sv - E0) / abs(E0) < 0.01  # <1% drift
        assert abs(E_y4 - E0) / abs(E0) < 0.01


class TestRobertsonKinetics:
    """Robertson chemical kinetics (extreme stiffness)."""

    def test_bdf_handles_robertson(self):
        """BDF should solve Robertson's stiff problem."""

        def f(t, y):
            y1, y2, y3 = y[0], y[1], y[2]
            dy1 = -0.04 * y1 + 1e4 * y2 * y3
            dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
            dy3 = 3e7 * y2**2
            return torch.stack([dy1, dy2, dy3])

        y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        # BDF should handle this extreme stiffness
        y_final, _ = bdf(f, y0, t_span=(0, 1e3), rtol=1e-4, atol=1e-8)

        # Conservation: y1 + y2 + y3 should stay constant
        assert not torch.isnan(y_final).any()
        total = y_final.sum().item()
        assert abs(total - 1.0) < 0.01


class TestNeuralODEBenchmark:
    """Neural ODE training benchmarks."""

    def test_reversible_heun_gradient(self):
        """Reversible Heun should compute accurate gradients."""
        W = torch.tensor(
            [[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )

        def f(t, y):
            return y @ W.T

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        y_final, _ = reversible_heun(f, y0, t_span=(0, 1), dt=0.01)
        loss = y_final.sum()
        loss.backward()

        assert W.grad is not None
        assert not torch.isnan(W.grad).any()


class TestSolverAccuracyComparison:
    """Compare accuracy across solver families."""

    def test_adaptive_vs_fixed_accuracy(self):
        """Adaptive solvers should achieve better accuracy."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        # Fixed step
        y_rk4, _ = runge_kutta_4(f, y0, t_span=(0, 1), dt=0.1)

        # Adaptive
        y_dp5, _ = dormand_prince_5(f, y0, t_span=(0, 1), rtol=1e-6, atol=1e-8)
        y_dop853, _ = dop853(f, y0, t_span=(0, 1), rtol=1e-6, atol=1e-8)

        error_rk4 = (y_rk4 - exact).abs().item()
        error_dp5 = (y_dp5 - exact).abs().item()
        error_dop853 = (y_dop853 - exact).abs().item()

        # Adaptive should be more accurate
        assert error_dp5 < error_rk4
        assert error_dop853 < error_rk4


class TestSolverRobustness:
    """Test solver robustness to difficult problems."""

    def test_near_singular(self):
        """Solvers should handle near-singular Jacobian."""

        def f(t, y):
            return torch.tensor(
                [y[1], -0.001 * y[0]], dtype=y.dtype, device=y.device
            )

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        # All should complete
        y_dp5, _ = dormand_prince_5(f, y0, t_span=(0, 100))
        assert not torch.isnan(y_dp5).any()

    def test_oscillatory(self):
        """Solvers should handle highly oscillatory problems."""

        def f(t, y):
            omega = 100.0
            return torch.stack([y[1], -(omega**2) * y[0]])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        # Need small step for oscillatory
        y_rk4, _ = runge_kutta_4(f, y0, t_span=(0, 1), dt=0.001)
        y_dp5, _ = dormand_prince_5(f, y0, t_span=(0, 1))

        assert not torch.isnan(y_rk4).any()
        assert not torch.isnan(y_dp5).any()

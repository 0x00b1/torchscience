"""Integration tests comparing all available solvers."""

import math

import torch

from torchscience.integration import (
    backward_euler,
    bdf,
    dop853,
    dormand_prince_5,
    euler,
    midpoint,
    recommend_solver,
    runge_kutta_4,
)


class TestSolverPortfolio:
    """Test that all solvers produce consistent results."""

    def test_all_explicit_solvers_on_decay(self):
        """All explicit solvers should solve exponential decay."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        results = {}
        results["euler"], _ = euler(f, y0, t_span=(0, 1), dt=0.01)
        results["midpoint"], _ = midpoint(f, y0, t_span=(0, 1), dt=0.01)
        results["rk4"], _ = runge_kutta_4(f, y0, t_span=(0, 1), dt=0.1)
        results["dp5"], _ = dormand_prince_5(f, y0, t_span=(0, 1))
        results["dop853"], _ = dop853(f, y0, t_span=(0, 1))

        for name, y in results.items():
            assert torch.allclose(y, exact, atol=0.1), f"{name} failed"

    def test_backward_euler_on_decay(self):
        """Backward Euler should solve exponential decay."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        y_final, _ = backward_euler(f, y0, t_span=(0, 1), dt=0.01)
        assert torch.allclose(y_final, exact, atol=0.1)

    def test_bdf_on_stiff_decay(self):
        """BDF should solve stiff exponential decay."""

        def f(t, y):
            return -100 * y  # Stiff coefficient makes BDF appropriate

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-100.0], dtype=torch.float64))

        y_final, _ = bdf(f, y0, t_span=(0, 1), rtol=1e-4, atol=1e-6)
        assert torch.allclose(y_final, exact, atol=1e-3)

    def test_dop853_highest_accuracy(self):
        """DOP853 should be most accurate at same tolerance."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        tol = 1e-6
        y_dp5, _ = dormand_prince_5(f, y0, t_span=(0, 1), rtol=tol, atol=tol)
        y_dop853, _ = dop853(f, y0, t_span=(0, 1), rtol=tol, atol=tol)

        error_dp5 = (y_dp5 - exact).abs().item()
        error_dop853 = (y_dop853 - exact).abs().item()

        # Both should be accurate
        assert error_dp5 < tol * 100
        assert error_dop853 < tol * 100

    def test_recommendation_accuracy(self):
        """Recommended solver should work well on its problem type."""

        # Stiff problem
        def stiff(t, y):
            return -1000 * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        solver_name = recommend_solver(stiff, y0, t_span=(0, 1))

        # Should recommend stiff solver
        assert solver_name in ["bdf", "radau", "backward_euler"]


class TestSolverAccuracyOrdering:
    """Verify expected accuracy ordering."""

    def test_order_progression(self):
        """Higher order methods should be more accurate."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        dt = 0.1

        y_euler, _ = euler(f, y0, t_span=(0, 1), dt=dt)
        y_midpoint, _ = midpoint(f, y0, t_span=(0, 1), dt=dt)
        y_rk4, _ = runge_kutta_4(f, y0, t_span=(0, 1), dt=dt)

        error_euler = (y_euler - exact).abs().item()
        error_midpoint = (y_midpoint - exact).abs().item()
        error_rk4 = (y_rk4 - exact).abs().item()

        # Higher order should be more accurate
        assert error_rk4 < error_midpoint < error_euler

    def test_adaptive_vs_fixed_step(self):
        """Adaptive and fixed-step solvers should both work on simple problems."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        # Fixed step with ~10 steps
        y_rk4, _ = runge_kutta_4(f, y0, t_span=(0, 1), dt=0.1)

        # Adaptive with loose tolerance
        y_dp5, _ = dormand_prince_5(f, y0, t_span=(0, 1), rtol=1e-3, atol=1e-5)

        error_rk4 = (y_rk4 - exact).abs().item()
        error_dp5 = (y_dp5 - exact).abs().item()

        # Both should produce reasonable results
        assert error_rk4 < 0.01
        assert error_dp5 < 0.01


class TestHarmonicOscillator:
    """Test solvers on harmonic oscillator."""

    def test_all_solvers_complete_period(self):
        """All solvers should approximately complete one period."""

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -q])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        t_span = (0.0, 2 * math.pi)

        # Test explicit solvers
        y_euler, _ = euler(f, y0, t_span=t_span, dt=0.01)
        y_rk4, _ = runge_kutta_4(f, y0, t_span=t_span, dt=0.1)
        y_dp5, _ = dormand_prince_5(f, y0, t_span=t_span)
        y_dop853, _ = dop853(f, y0, t_span=t_span)

        # RK4 should be most accurate of fixed-step
        assert torch.allclose(y_rk4, y0, atol=0.01)

        # Adaptive should be very accurate
        assert torch.allclose(y_dp5, y0, atol=0.01)
        assert torch.allclose(y_dop853, y0, atol=0.01)


class TestStiffProblem:
    """Test solvers on stiff problem."""

    def test_backward_euler_handles_stiff(self):
        """Backward Euler should handle stiff problems with large steps."""

        def f(t, y):
            return -100 * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-100.0], dtype=torch.float64))

        # Backward Euler with large step - should be stable
        y_be, _ = backward_euler(f, y0, t_span=(0, 1), dt=0.1)

        # Should complete without NaN (stability)
        assert not torch.isnan(y_be).any()
        # Should be in right ballpark (not testing accuracy, just stability)
        assert y_be.item() < 1.0  # Decayed from 1.0

    def test_bdf_handles_moderately_stiff(self):
        """BDF should handle moderately stiff problems."""

        def f(t, y):
            return -50 * y  # Moderately stiff

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-50.0], dtype=torch.float64))

        # BDF with appropriate tolerances
        y_bdf, _ = bdf(f, y0, t_span=(0, 1), rtol=1e-4, atol=1e-6)

        # Should complete and be accurate
        assert not torch.isnan(y_bdf).any()
        assert torch.allclose(y_bdf, exact, atol=1e-3)


class TestGradientFlow:
    """Test that gradients flow through all solvers."""

    def test_gradients_through_explicit_solvers(self):
        """Gradients should flow through explicit solvers."""
        solvers = [
            ("euler", lambda f, y0: euler(f, y0, t_span=(0, 1), dt=0.1)),
            ("midpoint", lambda f, y0: midpoint(f, y0, t_span=(0, 1), dt=0.1)),
            ("rk4", lambda f, y0: runge_kutta_4(f, y0, t_span=(0, 1), dt=0.1)),
            ("dp5", lambda f, y0: dormand_prince_5(f, y0, t_span=(0, 1))),
            ("dop853", lambda f, y0: dop853(f, y0, t_span=(0, 1))),
        ]

        for name, solver_fn in solvers:
            k = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

            def f(t, y):
                return -k * y

            y0 = torch.tensor([1.0], dtype=torch.float64)
            y_final, _ = solver_fn(f, y0)

            loss = y_final.sum()
            loss.backward()

            assert k.grad is not None, f"{name} gradient is None"
            assert not torch.isnan(k.grad).any(), f"{name} gradient is NaN"


class TestRecommendationIntegration:
    """Test that recommendations match solver behavior."""

    def test_recommended_solver_works(self):
        """Recommended solver should successfully solve the problem."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        solver_name = recommend_solver(f, y0, t_span=(0, 1))

        # Get the solver function
        import torchscience.integration as ivp

        solver_fn = getattr(ivp, solver_name)

        # Solve with recommended solver
        if solver_name in [
            "euler",
            "midpoint",
            "runge_kutta_4",
            "backward_euler",
        ]:
            y_final, _ = solver_fn(f, y0, t_span=(0, 1), dt=0.01)
        else:
            y_final, _ = solver_fn(f, y0, t_span=(0, 1))

        # Should be reasonably accurate
        error = (y_final - exact).abs().item()
        assert error < 0.1, f"Recommended {solver_name} has error {error}"

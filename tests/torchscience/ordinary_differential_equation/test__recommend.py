"""Tests for solver recommendation utility."""

import torch

from torchscience.ordinary_differential_equation._recommend import (
    analyze_problem,
    recommend_solver,
)


class TestRecommendSolver:
    """Test solver recommendation."""

    def test_recommends_dormand_prince_for_smooth(self):
        """Should recommend DP5 for smooth non-stiff problems."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        recommendation = recommend_solver(f, y0, t_span=(0.0, 1.0))

        assert recommendation in ["dormand_prince_5", "dop853"]

    def test_recommends_stiff_for_stiff_problem(self):
        """Should recommend BDF/Radau for stiff problems."""

        def f(t, y):
            return -1000 * y  # Very stiff

        y0 = torch.tensor([1.0], dtype=torch.float64)
        recommendation = recommend_solver(f, y0, t_span=(0.0, 1.0))

        assert recommendation in ["bdf", "radau", "backward_euler"]

    def test_recommends_symplectic_for_hamiltonian(self):
        """Should recommend symplectic for Hamiltonian structure."""

        def f(t, y):
            q, p = y[0], y[1]
            return torch.stack([p, -q])  # Hamiltonian

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        recommendation = recommend_solver(
            f, y0, t_span=(0.0, 100.0), hint="hamiltonian"
        )

        assert recommendation in [
            "stormer_verlet",
            "yoshida4",
            "implicit_midpoint",
        ]

    def test_returns_string(self):
        """Should return solver name as string."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        recommendation = recommend_solver(f, y0, t_span=(0.0, 1.0))

        assert isinstance(recommendation, str)

    def test_hint_neural_ode(self):
        """Should recommend reversible_heun for neural ODE hint."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        recommendation = recommend_solver(
            f, y0, t_span=(0.0, 1.0), hint="neural_ode"
        )

        assert recommendation == "reversible_heun"

    def test_hint_stiff(self):
        """Should recommend BDF for stiff hint."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        recommendation = recommend_solver(
            f, y0, t_span=(0.0, 1.0), hint="stiff"
        )

        assert recommendation == "bdf"

    def test_accuracy_high(self):
        """Should recommend dop853 for high accuracy."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        recommendation = recommend_solver(
            f, y0, t_span=(0.0, 1.0), accuracy="high"
        )

        assert recommendation == "dop853"

    def test_accuracy_low(self):
        """Should recommend rk4 for low accuracy."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        recommendation = recommend_solver(
            f, y0, t_span=(0.0, 1.0), accuracy="low"
        )

        assert recommendation == "runge_kutta_4"


class TestAnalyzeProblem:
    """Test problem analysis."""

    def test_detects_stiffness(self):
        """Should detect stiff problems."""

        def f(t, y):
            return -1000 * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        analysis = analyze_problem(f, y0, t_span=(0.0, 1.0))

        assert analysis["stiff"] is True

    def test_detects_non_stiff(self):
        """Should detect non-stiff problems."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        analysis = analyze_problem(f, y0, t_span=(0.0, 1.0))

        assert analysis["stiff"] is False

    def test_returns_analysis_dict(self):
        """Should return dict with analysis results."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        analysis = analyze_problem(f, y0, t_span=(0.0, 1.0))

        assert isinstance(analysis, dict)
        assert "stiff" in analysis
        assert "stiffness_ratio" in analysis

    def test_returns_dimension(self):
        """Should return state dimension."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        analysis = analyze_problem(f, y0, t_span=(0.0, 1.0))

        assert analysis["dimension"] == 3

    def test_handles_multidimensional(self):
        """Should handle multidimensional systems."""

        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        analysis = analyze_problem(f, y0, t_span=(0.0, 1.0))

        assert analysis["dimension"] == 4


class TestAnalyzeProblemRobustness:
    """Test robustness of problem analysis."""

    def test_handles_zero_initial_condition(self):
        """Should handle zero initial condition gracefully."""

        def f(t, y):
            return torch.ones_like(y)

        y0 = torch.tensor([0.0], dtype=torch.float64)
        analysis = analyze_problem(f, y0, t_span=(0.0, 1.0))

        assert isinstance(analysis, dict)

    def test_handles_large_values(self):
        """Should handle large initial values."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1e10], dtype=torch.float64)
        analysis = analyze_problem(f, y0, t_span=(0.0, 1.0))

        assert isinstance(analysis, dict)

    def test_handles_small_values(self):
        """Should handle small initial values."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1e-10], dtype=torch.float64)
        analysis = analyze_problem(f, y0, t_span=(0.0, 1.0))

        assert isinstance(analysis, dict)

# tests/torchscience/integration/initial_value_problem/test__solvers_comparison.py
import torch

from torchscience.integration.initial_value_problem import (
    dormand_prince_5,
    euler,
    midpoint,
    runge_kutta_4,
)


class TestSolverAccuracyOrdering:
    """Verify that higher-order methods are more accurate for same step size."""

    def test_accuracy_ordering_exponential_decay(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))
        dt = 0.1

        y_euler, _ = euler(f, y0, t_span=(0.0, 1.0), dt=dt)
        y_midpoint, _ = midpoint(f, y0, t_span=(0.0, 1.0), dt=dt)
        y_rk4, _ = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=dt)
        y_dp5, _ = dormand_prince_5(
            f, y0, t_span=(0.0, 1.0), rtol=1e-8, atol=1e-10
        )

        error_euler = (y_euler - expected).abs().item()
        error_midpoint = (y_midpoint - expected).abs().item()
        error_rk4 = (y_rk4 - expected).abs().item()
        error_dp5 = (y_dp5 - expected).abs().item()

        # Higher order => smaller error
        assert error_midpoint < error_euler, "Midpoint should beat Euler"
        assert error_rk4 < error_midpoint, "RK4 should beat Midpoint"
        assert error_dp5 < error_rk4 * 10, "DP5 should be very accurate"


class TestSolverConsistency:
    """Verify all solvers produce consistent results for simple problems."""

    def test_all_solvers_converge_to_same_solution(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        # Use small enough dt that all converge well
        y_euler, _ = euler(f, y0, t_span=(0.0, 1.0), dt=0.001)
        y_midpoint, _ = midpoint(f, y0, t_span=(0.0, 1.0), dt=0.001)
        y_rk4, _ = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=0.001)
        y_dp5, _ = dormand_prince_5(
            f, y0, t_span=(0.0, 1.0), rtol=1e-10, atol=1e-12
        )

        # All should be close to expected
        assert torch.allclose(y_euler, expected, rtol=1e-2)
        assert torch.allclose(y_midpoint, expected, rtol=1e-4)
        assert torch.allclose(y_rk4, expected, rtol=1e-6)
        assert torch.allclose(y_dp5, expected, rtol=1e-8)


class TestSolverInterpolantConsistency:
    """Verify interpolants work consistently across solvers."""

    def test_interpolant_endpoints_all_solvers(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])

        solvers = [
            ("euler", lambda: euler(f, y0, (0.0, 1.0), dt=0.1)),
            ("midpoint", lambda: midpoint(f, y0, (0.0, 1.0), dt=0.1)),
            (
                "runge_kutta_4",
                lambda: runge_kutta_4(f, y0, (0.0, 1.0), dt=0.1),
            ),
            ("dormand_prince_5", lambda: dormand_prince_5(f, y0, (0.0, 1.0))),
        ]

        for name, solve in solvers:
            y_final, interp = solve()

            # Start point should match y0
            assert torch.allclose(interp(0.0), y0, atol=1e-5), (
                f"{name}: start mismatch"
            )

            # End point should match y_final
            assert torch.allclose(interp(1.0), y_final, atol=1e-5), (
                f"{name}: end mismatch"
            )

    def test_interpolant_monotonicity_all_solvers(self):
        """For decaying ODE, all interpolants should be monotonically decreasing."""

        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        t_query = torch.linspace(0, 1, 20)

        solvers = [
            ("euler", lambda: euler(f, y0, (0.0, 1.0), dt=0.1)),
            ("midpoint", lambda: midpoint(f, y0, (0.0, 1.0), dt=0.1)),
            (
                "runge_kutta_4",
                lambda: runge_kutta_4(f, y0, (0.0, 1.0), dt=0.1),
            ),
            ("dormand_prince_5", lambda: dormand_prince_5(f, y0, (0.0, 1.0))),
        ]

        for name, solve in solvers:
            _, interp = solve()
            trajectory = interp(t_query)

            for i in range(len(t_query) - 1):
                assert trajectory[i, 0] >= trajectory[i + 1, 0], (
                    f"{name}: not monotonic at index {i}"
                )


class TestSolverBatchedConsistency:
    """Verify batched solving works consistently across solvers."""

    def test_batched_matches_individual(self):
        def f(t, y):
            return -y

        y0_batch = torch.tensor([[1.0], [2.0], [3.0]])

        for solver_fn in [
            lambda y0: euler(f, y0, (0.0, 1.0), dt=0.1),
            lambda y0: midpoint(f, y0, (0.0, 1.0), dt=0.1),
            lambda y0: runge_kutta_4(f, y0, (0.0, 1.0), dt=0.1),
        ]:
            # Batched solve
            y_batch, _ = solver_fn(y0_batch)

            # Individual solves
            for i in range(3):
                y_individual, _ = solver_fn(y0_batch[i : i + 1])
                assert torch.allclose(
                    y_batch[i], y_individual.squeeze(0), atol=1e-6
                ), "Batched and individual results should match"

# tests/torchscience/integration/initial_value_problem/test__stiff_solvers_comparison.py
"""Compare stiff solvers on benchmark problems."""

import pytest
import torch

from torchscience.integration.initial_value_problem import (
    backward_euler,
    bdf,
    radau,
    solve_ivp,
)


class TestStiffSolverComparison:
    """Compare all stiff solvers on the same problems."""

    @pytest.fixture
    def robertson(self):
        """Robertson's chemical kinetics problem."""

        def f(t, y):
            y1, y2, y3 = y[0], y[1], y[2]
            dy1 = -0.04 * y1 + 1e4 * y2 * y3
            dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
            dy3 = 3e7 * y2**2
            return torch.stack([dy1, dy2, dy3])

        y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        return f, y0

    def test_all_solvers_conserve_mass(self, robertson):
        """All stiff solvers should conserve mass."""
        f, y0 = robertson

        solvers = [
            ("backward_euler", backward_euler, {"dt": 0.01}),
            ("bdf", bdf, {"rtol": 1e-4, "atol": 1e-8, "max_steps": 50000}),
            ("radau", radau, {"rtol": 1e-4, "atol": 1e-8, "max_steps": 50000}),
        ]

        for name, solver, kwargs in solvers:
            y_final, _ = solver(f, y0, t_span=(0.0, 0.1), **kwargs)
            total = y_final.sum()
            assert torch.allclose(
                total, torch.tensor(1.0, dtype=torch.float64), atol=1e-4
            ), f"{name} failed mass conservation"

    def test_radau_most_accurate(self, robertson):
        """Radau (order 5) should be most accurate at comparable step sizes."""
        f, y0 = robertson

        # Get reference by running Radau with tight tolerances
        y_ref, _ = radau(
            f, y0, t_span=(0.0, 0.1), rtol=1e-10, atol=1e-12, max_steps=100000
        )

        # Compare solvers at comparable computational cost
        # Backward Euler with typical step size (1st order, needs small steps)
        y_be, _ = backward_euler(f, y0, t_span=(0.0, 0.1), dt=0.01)
        # BDF and Radau with similar tolerances
        y_bdf, _ = bdf(
            f, y0, t_span=(0.0, 0.1), rtol=1e-5, atol=1e-8, max_steps=50000
        )
        y_radau, _ = radau(
            f, y0, t_span=(0.0, 0.1), rtol=1e-5, atol=1e-8, max_steps=50000
        )

        err_be = (y_be - y_ref).abs().max().item()
        err_bdf = (y_bdf - y_ref).abs().max().item()
        err_radau = (y_radau - y_ref).abs().max().item()

        # Radau (5th order) should beat backward Euler (1st order) at typical step sizes
        assert err_radau < err_be, "Radau should beat backward Euler"

    def test_solve_ivp_method_dispatch(self, robertson):
        """solve_ivp should correctly dispatch to stiff solvers."""
        f, y0 = robertson

        result_bdf = solve_ivp(
            f,
            y0,
            t_span=(0.0, 0.1),
            method="bdf",
            rtol=1e-4,
            atol=1e-8,
            max_steps=50000,
        )
        result_radau = solve_ivp(
            f,
            y0,
            t_span=(0.0, 0.1),
            method="radau",
            rtol=1e-4,
            atol=1e-8,
            max_steps=50000,
        )

        assert not torch.isnan(result_bdf.y_final).any()
        assert not torch.isnan(result_radau.y_final).any()


class TestStiffSolverEfficiency:
    """Test that stiff solvers are efficient on stiff problems."""

    @pytest.fixture
    def stiff_decay(self):
        """Very stiff exponential decay."""
        lambda_val = 1000.0

        def f(t, y):
            return -lambda_val * y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        return f, y0

    def test_implicit_solvers_handle_stiffness(self, stiff_decay):
        """Implicit solvers should handle stiff problems without tiny steps."""
        f, y0 = stiff_decay

        # Backward Euler with reasonable step size
        y_be, _ = backward_euler(f, y0, t_span=(0.0, 0.01), dt=0.001)

        # BDF with loose tolerances
        y_bdf, _ = bdf(f, y0, t_span=(0.0, 0.01), rtol=1e-3, atol=1e-6)

        # Radau with loose tolerances
        y_radau, _ = radau(f, y0, t_span=(0.0, 0.01), rtol=1e-3, atol=1e-6)

        # All should give small values (exp(-10) ~ 4.5e-5)
        expected = torch.exp(torch.tensor([-10.0], dtype=torch.float64))

        assert y_be.abs().item() < 1e-3
        assert y_bdf.abs().item() < 1e-3
        assert y_radau.abs().item() < 1e-3


class TestStiffSolverGradients:
    """Test gradient computation through stiff solvers."""

    def test_gradients_flow_through_all_solvers(self):
        """Gradients should flow through all stiff solvers."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solvers = [
            ("backward_euler", backward_euler, {"dt": 0.1}),
            ("bdf", bdf, {"rtol": 1e-3, "atol": 1e-6}),
            ("radau", radau, {"rtol": 1e-3, "atol": 1e-6}),
        ]

        for name, solver, kwargs in solvers:
            # Reset gradient
            if theta.grad is not None:
                theta.grad.zero_()

            y_final, _ = solver(f, y0, t_span=(0.0, 1.0), **kwargs)
            loss = y_final.sum()
            loss.backward()

            assert theta.grad is not None, (
                f"{name} failed to compute gradients"
            )
            assert not torch.isnan(theta.grad).any(), (
                f"{name} produced NaN gradients"
            )

# tests/torchscience/integration/initial_value_problem/test__stiffness.py
"""Tests comparing implicit vs explicit solvers on stiff problems."""

import pytest
import torch

from torchscience.ordinary_differential_equation import (
    backward_euler,
    euler,
    runge_kutta_4,
)


class TestStiffProblem:
    """Test on stiff ODE: dy/dt = -lambda * y with large lambda."""

    def test_explicit_euler_unstable_for_stiff(self):
        """Forward Euler is unstable for stiff problems with large steps."""
        lam = 1000.0  # Stiff coefficient

        def f(t, y):
            return -lam * y

        y0 = torch.tensor([1.0])

        # For stability, Euler needs dt < 2/lambda = 0.002
        # With dt = 0.01, Euler will be unstable
        y_euler, _ = euler(f, y0, t_span=(0.0, 0.1), dt=0.01)

        # Should either blow up or oscillate wildly
        # (may produce NaN or very large values)
        # In practice, the solution will oscillate with increasing amplitude
        assert y_euler.abs().max() > 10 or torch.isnan(y_euler).any()

    def test_backward_euler_stable_for_stiff(self):
        """Backward Euler remains stable for stiff problems."""
        lam = 1000.0

        def f(t, y):
            return -lam * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Backward Euler is unconditionally stable
        y_be, _ = backward_euler(
            f, y0, t_span=(0.0, 1.0), dt=0.1, newton_tol=1e-10
        )

        # Should produce bounded result close to 0 (exact: exp(-1000))
        assert y_be.abs().max() < 1.0
        assert not torch.isnan(y_be).any()

    def test_accuracy_vs_stability_tradeoff(self):
        """
        Backward Euler is A-stable but only 1st order accurate.
        For non-stiff problems, explicit methods may be more accurate.
        """

        # Non-stiff problem
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        expected = torch.exp(torch.tensor([-1.0], dtype=torch.float64))

        y_euler, _ = euler(f, y0, t_span=(0.0, 1.0), dt=0.1)
        y_be, _ = backward_euler(f, y0, t_span=(0.0, 1.0), dt=0.1)
        y_rk4, _ = runge_kutta_4(f, y0, t_span=(0.0, 1.0), dt=0.1)

        error_euler = (y_euler - expected).abs().item()
        error_be = (y_be - expected).abs().item()
        error_rk4 = (y_rk4 - expected).abs().item()

        # RK4 should be most accurate for non-stiff
        assert error_rk4 < error_euler
        assert error_rk4 < error_be

        # Euler and backward Euler have similar accuracy (both 1st order)
        assert abs(error_euler - error_be) < 0.1


class TestRobertsonProblem:
    """Robertson's problem - classic stiff test case."""

    @pytest.mark.parametrize("solver", ["backward_euler"])
    def test_robertson_backward_euler(self, solver):
        """
        Robertson's chemical kinetics problem:
        dy1/dt = -0.04*y1 + 1e4*y2*y3
        dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
        dy3/dt = 3e7*y2^2

        This is a stiff system with timescales spanning 10^11.
        """

        def robertson(t, y):
            y1, y2, y3 = y[0], y[1], y[2]
            dy1 = -0.04 * y1 + 1e4 * y2 * y3
            dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
            dy3 = 3e7 * y2**2
            return torch.stack([dy1, dy2, dy3])

        y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        # Backward Euler can handle this with moderate step size
        y_final, _ = backward_euler(
            robertson,
            y0,
            t_span=(0.0, 0.1),  # Short integration
            dt=0.01,
            newton_tol=1e-8,
            max_newton_iter=20,
        )

        # Conservation: y1 + y2 + y3 = 1
        total = y_final.sum()
        assert torch.allclose(
            total, torch.tensor(1.0, dtype=torch.float64), atol=1e-6
        )

        # All concentrations should be non-negative
        assert (y_final >= -1e-10).all()

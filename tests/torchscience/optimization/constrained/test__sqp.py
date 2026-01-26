import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.constrained._sqp import sqp


class TestSQP:
    def test_unconstrained_quadratic(self):
        """Without constraints, should minimize f(x) = ||x||^2."""

        def objective(x):
            return torch.sum(x**2)

        result = sqp(objective, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_equality_constraint(self):
        """Minimize x^2 + y^2 subject to x + y = 1."""

        def objective(x):
            return torch.sum(x**2)

        def eq_constraints(x):
            return x.sum() - 1.0

        result = sqp(
            objective,
            torch.tensor([0.6, 0.4]),
            eq_constraints=eq_constraints,
        )
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(result.x, expected, atol=1e-3, rtol=1e-3)

    def test_inequality_constraint(self):
        """Minimize -x subject to x <= 2."""

        def objective(x):
            return -x.sum()

        def ineq_constraints(x):
            return x - 2.0

        result = sqp(
            objective,
            torch.tensor([0.0]),
            ineq_constraints=ineq_constraints,
        )
        expected = torch.tensor([2.0])
        torch.testing.assert_close(result.x, expected, atol=1e-2, rtol=1e-2)

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def objective(x):
            return torch.sum(x**2)

        result = sqp(objective, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)

    def test_convergence_flag(self):
        """Test convergence flag."""

        def objective(x):
            return torch.sum(x**2)

        result = sqp(objective, torch.tensor([1.0]))
        assert result.converged.item() is True


class TestSQPDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def objective(x):
            return torch.sum(x**2)

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = sqp(objective, x0)
        assert result.x.dtype == dtype

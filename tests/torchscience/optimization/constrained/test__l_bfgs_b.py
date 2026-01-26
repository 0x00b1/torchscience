import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.constrained._l_bfgs_b import l_bfgs_b


class TestLBFGSB:
    def test_unconstrained(self):
        """Without bounds, should behave like L-BFGS."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_lower_bound_active(self):
        """Minimize x^2 with lower bound x >= 2."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(
            f,
            torch.tensor([5.0]),
            lower=torch.tensor([2.0]),
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([2.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_upper_bound_active(self):
        """Minimize (x-5)^2 with upper bound x <= 3."""

        def f(x):
            return ((x - 5) ** 2).sum()

        result = l_bfgs_b(
            f,
            torch.tensor([0.0]),
            upper=torch.tensor([3.0]),
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([3.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_box_bounds(self):
        """Minimize (x-5)^2 + (y+3)^2 with 0 <= x <= 3, -1 <= y <= 1."""

        def f(x):
            return (x[0] - 5) ** 2 + (x[1] + 3) ** 2

        result = l_bfgs_b(
            f,
            torch.tensor([1.0, 0.0]),
            lower=torch.tensor([0.0, -1.0]),
            upper=torch.tensor([3.0, 1.0]),
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([3.0, -1.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_inactive_bounds(self):
        """Bounds that don't affect the solution."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(
            f,
            torch.tensor([1.0, 1.0]),
            lower=torch.tensor([-10.0, -10.0]),
            upper=torch.tensor([10.0, 10.0]),
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(f, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.num_iterations is not None
        assert result.fun is not None

    def test_convergence_flag(self):
        """Test convergence flag."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(f, torch.tensor([1.0]))
        assert result.converged.item() is True

    def test_rosenbrock_bounded(self):
        """Rosenbrock with bounds."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = l_bfgs_b(
            rosenbrock,
            torch.tensor([0.0, 0.0]),
            lower=torch.tensor([-2.0, -2.0]),
            upper=torch.tensor([2.0, 2.0]),
            maxiter=200,
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 1.0]),
            atol=1e-2,
            rtol=1e-2,
        )


class TestLBFGSBAutograd:
    def test_implicit_diff(self):
        """Test implicit differentiation with active bound."""
        bound = torch.tensor([2.0], requires_grad=True)

        def f(x):
            return (x**2).sum()

        result = l_bfgs_b(
            f,
            torch.tensor([5.0]),
            lower=bound,
        )
        result.x.sum().backward()
        # x* = bound, so dx*/dbound = 1
        torch.testing.assert_close(
            bound.grad,
            torch.tensor([1.0]),
            atol=1e-2,
            rtol=1e-2,
        )


class TestLBFGSBDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def f(x):
            return (x**2).sum()

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = l_bfgs_b(f, x0)
        assert result.x.dtype == dtype

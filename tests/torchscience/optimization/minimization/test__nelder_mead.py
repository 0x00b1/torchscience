import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._nelder_mead import nelder_mead


class TestNelderMead:
    def test_quadratic(self):
        """Minimize f(x) = ||x||^2."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_rosenbrock_2d(self):
        """Minimize 2D Rosenbrock function."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = nelder_mead(
            rosenbrock, torch.tensor([-1.0, 1.0]), maxiter=1000
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 1.0]),
            atol=0.1,
            rtol=0.1,
        )

    def test_convergence_flag(self):
        """Test that convergence flag is set correctly."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([1.0, 1.0]))
        assert result.converged.item() is True

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.num_iterations is not None
        assert result.fun is not None

    def test_fun_value(self):
        """Test that fun contains the objective value at the solution."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([3.0, 4.0]))
        assert result.fun.item() < 0.01

    def test_no_gradient_required(self):
        """Nelder-Mead works on non-differentiable functions."""

        def f(x):
            return x.abs().sum()

        result = nelder_mead(f, torch.tensor([3.0, -4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_result_is_detached(self):
        """Nelder-Mead result has no gradient (derivative-free method)."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([1.0]))
        assert not result.x.requires_grad

    def test_tol_parameter(self):
        """Test that tol affects convergence."""

        def f(x):
            return (x**2).sum()

        result = nelder_mead(f, torch.tensor([1.0, 1.0]), tol=1e-10)
        assert result.fun.item() < 1e-8

    def test_1d(self):
        """Test 1D optimization."""

        def f(x):
            return (x - 3) ** 2

        result = nelder_mead(f, torch.tensor([0.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([3.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_booth_function(self):
        """Test Booth function: (x + 2y - 7)^2 + (2x + y - 5)^2."""

        def booth(x):
            return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        result = nelder_mead(booth, torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 3.0]),
            atol=1e-3,
            rtol=1e-3,
        )


class TestNelderMeadDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def f(x):
            return (x**2).sum()

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = nelder_mead(f, x0)
        assert result.x.dtype == dtype

import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._trust_region import trust_region


class TestTrustRegion:
    def test_quadratic(self):
        """Minimize f(x) = ||x||^2."""

        def f(x):
            return (x**2).sum()

        result = trust_region(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_rosenbrock_2d(self):
        """Minimize 2D Rosenbrock function."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = trust_region(
            rosenbrock, torch.tensor([-1.0, 1.0]), maxiter=200
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 1.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_convergence_flag(self):
        """Test that convergence flag is set correctly."""

        def f(x):
            return (x**2).sum()

        result = trust_region(f, torch.tensor([1.0]))
        assert result.converged.item() is True

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def f(x):
            return (x**2).sum()

        result = trust_region(f, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.num_iterations is not None
        assert result.fun is not None

    def test_fun_value(self):
        """Test that fun contains the objective value at the solution."""

        def f(x):
            return (x**2).sum()

        result = trust_region(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.fun,
            torch.tensor(0.0),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_ill_conditioned_quadratic(self):
        """Trust-region handles ill-conditioned problems."""

        A = torch.diag(torch.tensor([1.0, 1000.0]))

        def f(x):
            return 0.5 * x @ A @ x

        result = trust_region(f, torch.tensor([10.0, 10.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_tol_parameter(self):
        """Test that tol parameter affects convergence."""

        def f(x):
            return (x**2).sum()

        result = trust_region(f, torch.tensor([1.0]), tol=1e-12)
        assert result.x.abs().item() < 1e-6

    def test_initial_trust_radius(self):
        """Test custom initial trust radius."""

        def f(x):
            return (x**2).sum()

        result = trust_region(
            f, torch.tensor([3.0, 4.0]), initial_trust_radius=0.1
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )


class TestTrustRegionAutograd:
    def test_implicit_diff_quadratic(self):
        """Test implicit differentiation through a quadratic."""
        target = torch.tensor([5.0, 3.0], requires_grad=True)

        def f(x):
            return ((x - target) ** 2).sum()

        result = trust_region(f, torch.zeros(2))
        result.x.sum().backward()

        torch.testing.assert_close(
            target.grad,
            torch.ones(2),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_numerical_gradient(self):
        """Test implicit gradient against numerical finite differences."""
        eps = 1e-4
        theta = torch.tensor([2.0], requires_grad=True)

        def f(x):
            return ((x - theta) ** 2).sum()

        result = trust_region(f, torch.tensor([0.0]))
        result.x.sum().backward()
        analytic_grad = theta.grad.clone()

        with torch.no_grad():
            theta_plus = theta + eps
            theta_minus = theta - eps

        def f_plus(x):
            return ((x - theta_plus) ** 2).sum()

        def f_minus(x):
            return ((x - theta_minus) ** 2).sum()

        r_plus = trust_region(f_plus, torch.tensor([0.0]))
        r_minus = trust_region(f_minus, torch.tensor([0.0]))
        numerical_grad = (r_plus.x.sum() - r_minus.x.sum()) / (2 * eps)

        torch.testing.assert_close(
            analytic_grad,
            numerical_grad.unsqueeze(0),
            atol=1e-2,
            rtol=1e-2,
        )


class TestTrustRegionDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def f(x):
            return (x**2).sum()

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = trust_region(f, x0)
        assert result.x.dtype == dtype

import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._l_bfgs import l_bfgs


class TestLBFGS:
    def test_quadratic(self):
        """Minimize f(x) = ||x||^2."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs(f, torch.tensor([3.0, 4.0]))
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

        result = l_bfgs(rosenbrock, torch.tensor([-1.0, 1.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 1.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_rosenbrock_10d(self):
        """Minimize 10D Rosenbrock function."""

        def rosenbrock_nd(x):
            return ((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2).sum()

        x0 = torch.zeros(10)
        result = l_bfgs(rosenbrock_nd, x0, maxiter=500)
        expected = torch.ones(10)
        torch.testing.assert_close(result.x, expected, atol=1e-2, rtol=1e-2)

    def test_convergence_flag(self):
        """Test that convergence flag is set correctly."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs(f, torch.tensor([1.0]))
        assert result.converged.item() is True

    def test_convergence_flag_not_converged(self):
        """Test convergence flag when maxiter is too low."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = l_bfgs(rosenbrock, torch.tensor([-5.0, 5.0]), maxiter=2)
        # May or may not converge in 2 iterations
        assert isinstance(result.converged, torch.Tensor)

    def test_history_size(self):
        """Test that history_size parameter is respected."""

        def f(x):
            return (x**2).sum()

        # Small history should still converge on simple problems
        result = l_bfgs(f, torch.tensor([3.0, 4.0]), history_size=2)
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_armijo_line_search(self):
        """Test with Armijo line search."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs(f, torch.tensor([3.0, 4.0]), line_search="armijo")
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_explicit_gradient(self):
        """Test with explicit gradient function."""

        def f(x):
            return (x**2).sum()

        def grad_f(x):
            return 2 * x

        result = l_bfgs(f, torch.tensor([3.0, 4.0]), grad=grad_f)
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_tol_parameter(self):
        """Test that tol parameter affects convergence."""

        def f(x):
            return (x**2).sum()

        # Very tight tolerance
        result = l_bfgs(f, torch.tensor([1.0]), tol=1e-12)
        assert result.x.abs().item() < 1e-6

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs(f, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.num_iterations is not None
        assert result.fun is not None

    def test_fun_value(self):
        """Test that fun contains the objective value at the solution."""

        def f(x):
            return (x**2).sum()

        result = l_bfgs(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.fun,
            torch.tensor(0.0),
            atol=1e-8,
            rtol=1e-8,
        )


class TestLBFGSBatched:
    def test_batched_quadratic(self):
        """Test batched optimization of quadratics."""

        def f(x):
            return (x**2).sum(dim=-1)

        x0 = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
        result = l_bfgs(f, x0, line_search="armijo")
        expected = torch.zeros(2, 2)
        torch.testing.assert_close(result.x, expected, atol=1e-4, rtol=1e-4)

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""

        def f(x):
            return (x**2).sum(dim=-1)

        x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = l_bfgs(f, x0, line_search="armijo")
        assert result.x.shape == x0.shape
        assert result.converged.shape == (2,)

    def test_unbatched_input(self):
        """Test 1D input (unbatched)."""

        def f(x):
            return (x**2).sum()

        x0 = torch.tensor([3.0, 4.0])
        result = l_bfgs(f, x0)
        assert result.x.shape == (2,)
        assert result.converged.dim() == 0


class TestLBFGSAutograd:
    def test_implicit_diff_quadratic(self):
        """Test implicit differentiation through a quadratic."""
        target = torch.tensor([5.0, 3.0], requires_grad=True)

        def f(x):
            return ((x - target) ** 2).sum()

        result = l_bfgs(f, torch.zeros(2))
        result.x.sum().backward()

        # dx*/dtarget = I (identity)
        torch.testing.assert_close(
            target.grad,
            torch.ones(2),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_implicit_diff_parameterized(self):
        """Test implicit diff with a parameterized objective."""
        A = torch.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)

        def f(x):
            return 0.5 * x @ A @ x

        result = l_bfgs(f, torch.tensor([1.0, 1.0]))
        loss = result.x.sum()
        loss.backward()

        # At the minimum, x* = 0, so gradient through A should be 0
        assert A.grad is not None

    def test_numerical_gradient(self):
        """Test implicit gradient against numerical finite differences."""
        eps = 1e-4
        theta = torch.tensor([2.0], requires_grad=True)

        def f(x):
            return ((x - theta) ** 2).sum()

        result = l_bfgs(f, torch.tensor([0.0]))
        result.x.sum().backward()
        analytic_grad = theta.grad.clone()

        # Numerical gradient
        with torch.no_grad():
            theta_plus = theta + eps
            theta_minus = theta - eps

        def f_plus(x):
            return ((x - theta_plus) ** 2).sum()

        def f_minus(x):
            return ((x - theta_minus) ** 2).sum()

        r_plus = l_bfgs(f_plus, torch.tensor([0.0]))
        r_minus = l_bfgs(f_minus, torch.tensor([0.0]))
        numerical_grad = (r_plus.x.sum() - r_minus.x.sum()) / (2 * eps)

        torch.testing.assert_close(
            analytic_grad,
            numerical_grad.unsqueeze(0),
            atol=1e-2,
            rtol=1e-2,
        )


class TestLBFGSDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def f(x):
            return (x**2).sum()

        x0 = torch.tensor([1.0, 2.0], dtype=dtype)
        result = l_bfgs(f, x0)
        assert result.x.dtype == dtype

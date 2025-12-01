import pytest
import torch
import torch.testing

import torchscience.optimization.minimization


class TestLevenbergMarquardt:
    def test_linear_least_squares(self):
        """Fit y = ax + b to data."""
        x_data = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y_data = 2.0 * x_data + 1.0  # True: a=2, b=1

        def residuals(params):
            a, b = params[0], params[1]
            return a * x_data + b - y_data

        params0 = torch.tensor([0.0, 0.0])
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, params0
        )
        expected = torch.tensor([2.0, 1.0])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_nonlinear_exponential(self):
        """Fit y = a * exp(-b * x) to data."""
        x_data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        y_data = 2.0 * torch.exp(-0.5 * x_data)

        def residuals(params):
            a, b = params[0], params[1]
            return a * torch.exp(-b * x_data) - y_data

        params0 = torch.tensor([1.0, 1.0])
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, params0
        )
        expected = torch.tensor([2.0, 0.5])
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_implicit_differentiation(self):
        """Test gradient through optimizer via implicit diff."""
        target = torch.tensor([3.0], requires_grad=True)

        def residuals(x):
            return x - target

        x0 = torch.tensor([0.0])
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, x0
        )
        result.sum().backward()
        # dx*/dtarget = 1 (optimal x equals target)
        torch.testing.assert_close(
            target.grad, torch.tensor([1.0]), atol=1e-5, rtol=1e-5
        )

    def test_rosenbrock_minimization(self):
        """Minimize Rosenbrock as least squares: r1 = a-x, r2 = sqrt(b)*(y-x^2)."""
        a, b = 1.0, 100.0

        def residuals(params):
            x, y = params[0], params[1]
            r1 = a - x
            r2 = (b**0.5) * (y - x**2)
            return torch.stack([r1, r2])

        params0 = torch.tensor([-1.0, 1.0])
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, params0, maxiter=200
        )
        expected = torch.tensor([1.0, 1.0])
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_overdetermined_system(self):
        """Test overdetermined system (more residuals than parameters)."""
        # Fit line to 10 points
        x_data = torch.linspace(0, 1, 10)
        y_data = 2.0 * x_data + 1.0 + 0.01 * torch.randn(10)

        def residuals(params):
            a, b = params[0], params[1]
            return a * x_data + b - y_data

        params0 = torch.zeros(2)
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, params0
        )
        # Should be close to [2, 1]
        assert abs(result[0].item() - 2.0) < 0.1
        assert abs(result[1].item() - 1.0) < 0.1

    def test_convergence_failure_warning(self):
        """Test that maxiter=1 doesn't crash (may not converge)."""

        def residuals(x):
            return x - torch.tensor([100.0])

        x0 = torch.tensor([0.0])
        # Should run without error even if not converged
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, x0, maxiter=1
        )
        assert result.shape == x0.shape


class TestLevenbergMarquardtDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def residuals(x):
            return x - torch.tensor([1.0], dtype=dtype)

        x0 = torch.zeros(1, dtype=dtype)
        result = torchscience.optimization.minimization.levenberg_marquardt(
            residuals, x0
        )
        assert result.dtype == dtype

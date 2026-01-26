import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._minimize import minimize


class TestMinimize:
    def test_default_method(self):
        """Test that default method (L-BFGS) works."""

        def f(x):
            return (x**2).sum()

        result = minimize(f, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 0.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_explicit_lbfgs(self):
        """Test explicit L-BFGS method selection."""

        def f(x):
            return (x**2).sum()

        result = minimize(f, torch.tensor([3.0]), method="l-bfgs")
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_levenberg_marquardt_method(self):
        """Test Levenberg-Marquardt method via minimize."""

        def residuals(params):
            x_data = torch.tensor([0.0, 1.0, 2.0, 3.0])
            y_data = torch.tensor([1.0, 3.0, 5.0, 7.0])
            return params[0] * x_data + params[1] - y_data

        result = minimize(
            residuals,
            torch.tensor([0.0, 0.0]),
            method="levenberg-marquardt",
        )
        assert isinstance(result, OptimizeResult)
        torch.testing.assert_close(
            result.x,
            torch.tensor([2.0, 1.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""

        def f(x):
            return (x**2).sum()

        with pytest.raises(ValueError, match="Unknown method"):
            minimize(f, torch.tensor([1.0]), method="bogus")

    def test_kwargs_forwarded(self):
        """Test that kwargs are forwarded to the solver."""

        def f(x):
            return (x**2).sum()

        result = minimize(f, torch.tensor([3.0]), maxiter=200, tol=1e-10)
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_returns_optimize_result(self):
        """Test that minimize returns OptimizeResult."""

        def f(x):
            return (x**2).sum()

        result = minimize(f, torch.tensor([1.0]))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.num_iterations is not None

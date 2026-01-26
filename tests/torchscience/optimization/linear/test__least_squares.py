import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.linear._least_squares import least_squares


class TestLeastSquares:
    def test_linear_system(self):
        """Solve a linear least squares problem."""

        def residuals(x):
            # ||Ax - b||^2 where A = I, b = [1, 2]
            return x - torch.tensor([1.0, 2.0])

        result = least_squares(residuals, torch.zeros(2))
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 2.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_nonlinear(self):
        """Solve a nonlinear least squares problem."""

        def residuals(x):
            # Fit y = a * exp(b * t) to data
            t = torch.tensor([0.0, 1.0, 2.0, 3.0])
            y = torch.tensor([1.0, 2.7, 7.4, 20.1])
            return x[0] * torch.exp(x[1] * t) - y

        result = least_squares(residuals, torch.tensor([1.0, 0.5]))
        # Should converge to approximately a=1, b=1
        assert result.fun.item() < 0.1

    def test_result_type(self):
        """Test that result is an OptimizeResult."""

        def residuals(x):
            return x - torch.tensor([1.0])

        result = least_squares(residuals, torch.zeros(1))
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.fun is not None

    def test_bounded(self):
        """Test bounded least squares."""

        def residuals(x):
            return x - torch.tensor([5.0])

        result = least_squares(
            residuals,
            torch.tensor([0.0]),
            bounds=(torch.tensor([0.0]), torch.tensor([3.0])),
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([3.0]),
            atol=1e-2,
            rtol=1e-2,
        )


class TestLeastSquaresDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def residuals(x):
            return x

        x0 = torch.tensor([1.0], dtype=dtype)
        result = least_squares(residuals, x0)
        assert result.x.dtype == dtype

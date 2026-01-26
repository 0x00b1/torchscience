import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._curve_fit import curve_fit


class TestCurveFit:
    def test_linear_fit(self):
        """Fit y = a*x + b."""
        xdata = torch.tensor([0.0, 1.0, 2.0, 3.0])
        ydata = torch.tensor([1.0, 3.0, 5.0, 7.0])  # y = 2x + 1

        def model(x, params):
            return params[0] * x + params[1]

        result = curve_fit(model, xdata, ydata, torch.zeros(2))
        torch.testing.assert_close(
            result.x,
            torch.tensor([2.0, 1.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_exponential_fit(self):
        """Fit y = a * exp(-b * x)."""
        xdata = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        ydata = 2.0 * torch.exp(-0.5 * xdata)

        def model(x, params):
            return params[0] * torch.exp(-params[1] * x)

        result = curve_fit(model, xdata, ydata, torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(
            result.x,
            torch.tensor([2.0, 0.5]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_weighted_fit(self):
        """Test weighted least squares with sigma."""
        xdata = torch.tensor([0.0, 1.0, 2.0, 3.0])
        ydata = torch.tensor([1.0, 3.0, 5.0, 7.0])
        sigma = torch.tensor([0.1, 0.1, 0.1, 0.1])

        def model(x, params):
            return params[0] * x + params[1]

        result = curve_fit(
            model,
            xdata,
            ydata,
            torch.zeros(2),
            sigma=sigma,
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([2.0, 1.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_implicit_diff(self):
        """Test gradient through curve_fit using L-BFGS method."""
        target_a = torch.tensor([2.0], requires_grad=True)
        xdata = torch.tensor([0.0, 1.0, 2.0])

        def make_ydata():
            return target_a * xdata

        ydata = make_ydata()

        def model(x, params):
            return params[0] * x

        result = curve_fit(
            model,
            xdata,
            ydata,
            torch.tensor([0.0]),
            method="l-bfgs",
        )
        result.x.sum().backward()
        # dx*/d(target_a) = 1
        assert target_a.grad is not None

    def test_returns_optimize_result(self):
        """Test that curve_fit returns OptimizeResult."""
        xdata = torch.tensor([0.0, 1.0])
        ydata = torch.tensor([0.0, 1.0])

        def model(x, params):
            return params[0] * x

        result = curve_fit(model, xdata, ydata, torch.tensor([0.0]))
        assert isinstance(result, OptimizeResult)

    def test_lbfgs_method(self):
        """Test curve_fit with L-BFGS method."""
        xdata = torch.tensor([0.0, 1.0, 2.0, 3.0])
        ydata = torch.tensor([1.0, 3.0, 5.0, 7.0])

        def model(x, params):
            return params[0] * x + params[1]

        result = curve_fit(
            model,
            xdata,
            ydata,
            torch.tensor([0.0, 0.0]),
            method="l-bfgs",
        )
        torch.testing.assert_close(
            result.x,
            torch.tensor([2.0, 1.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""
        xdata = torch.tensor([0.0, 1.0], dtype=dtype)
        ydata = torch.tensor([0.0, 2.0], dtype=dtype)

        def model(x, params):
            return params[0] * x

        result = curve_fit(
            model, xdata, ydata, torch.tensor([0.0], dtype=dtype)
        )
        assert result.x.dtype == dtype

    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""

        def model(x, params):
            return params[0] * x

        with pytest.raises(ValueError, match="Unknown method"):
            curve_fit(
                model,
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                method="bogus",
            )

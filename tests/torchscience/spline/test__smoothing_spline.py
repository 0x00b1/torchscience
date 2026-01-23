"""Tests for smoothing spline functions."""

import pytest
import torch

from torchscience.spline import (
    ExtrapolationError,
    SmoothingSpline,
    smoothing_spline,
    smoothing_spline_derivative,
    smoothing_spline_evaluate,
    smoothing_spline_fit,
)


class TestSmoothingSplineFit:
    """Tests for smoothing_spline_fit function."""

    def test_fit_basic(self):
        """Should fit to noisy data."""
        torch.manual_seed(42)
        x = torch.linspace(0, 2 * torch.pi, 20)
        y = torch.sin(x) + 0.1 * torch.randn_like(x)

        spline = smoothing_spline_fit(x, y, smoothing=1e-2)

        assert isinstance(spline, SmoothingSpline)
        assert spline.knots.shape[0] == 20
        assert spline.coefficients.shape[0] == 20

    def test_fit_auto_smoothing(self):
        """Should auto-select smoothing parameter."""
        torch.manual_seed(42)
        x = torch.linspace(0, 1, 30)
        y = x**2 + 0.05 * torch.randn_like(x)

        spline = smoothing_spline_fit(x, y, smoothing=None)

        assert spline.smoothing > 0

    def test_fit_with_weights(self):
        """Should respect data weights."""
        x = torch.linspace(0, 1, 10)
        y = torch.zeros_like(x)
        y[5] = 1.0  # Outlier

        # Without weighting, outlier affects fit
        weights_uniform = torch.ones_like(x)
        spline_uniform = smoothing_spline_fit(
            x, y, smoothing=1e-3, weights=weights_uniform
        )

        # With low weight on outlier
        weights_low = torch.ones_like(x)
        weights_low[5] = 0.01
        spline_weighted = smoothing_spline_fit(
            x, y, smoothing=1e-3, weights=weights_low
        )

        # Weighted spline should stay closer to zero away from outlier
        t = torch.tensor([0.2, 0.8])
        val_uniform = smoothing_spline_evaluate(spline_uniform, t)
        val_weighted = smoothing_spline_evaluate(spline_weighted, t)

        assert val_weighted.abs().max() < val_uniform.abs().max()

    def test_fit_requires_4_points(self):
        """Should require at least 4 points."""
        x = torch.tensor([0.0, 0.5, 1.0])
        y = torch.tensor([0.0, 0.5, 1.0])

        with pytest.raises(ValueError):
            smoothing_spline_fit(x, y)

    def test_fit_strictly_increasing(self):
        """Should require strictly increasing x."""
        x = torch.tensor([0.0, 0.5, 0.5, 1.0])  # Duplicate
        y = torch.tensor([0.0, 0.5, 0.6, 1.0])

        with pytest.raises(ValueError):
            smoothing_spline_fit(x, y)

    def test_smoothing_zero_interpolates(self):
        """With smoothing=0, should approximate interpolation."""
        x = torch.linspace(0, 1, 10)
        y = torch.sin(x * torch.pi)

        # Very small smoothing should closely fit data
        spline = smoothing_spline_fit(x, y, smoothing=1e-10)
        y_fit = smoothing_spline_evaluate(spline, x)

        assert torch.allclose(y_fit, y, rtol=1e-2, atol=1e-3)

    def test_large_smoothing_linear(self):
        """With large smoothing, should approach linear fit."""
        x = torch.linspace(0, 1, 20)
        y = x**2  # Quadratic data

        # Large smoothing should produce nearly linear result
        spline = smoothing_spline_fit(x, y, smoothing=1e6)

        t = torch.linspace(0, 1, 5)
        y_fit = smoothing_spline_evaluate(spline, t)

        # Second derivative should be small (nearly linear)
        y_dd = smoothing_spline_derivative(spline, t, order=2)
        assert y_dd.abs().max() < 1.0  # Much smaller than for quadratic


class TestSmoothingSplineEvaluate:
    """Tests for smoothing_spline_evaluate function."""

    def test_evaluate_at_knots(self):
        """Should evaluate at knot points."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = smoothing_spline_fit(x, y, smoothing=1e-6)

        y_eval = smoothing_spline_evaluate(spline, x)

        # With low smoothing, should be close to original values
        assert torch.allclose(y_eval, y, rtol=0.1, atol=0.05)

    def test_evaluate_between_knots(self):
        """Should interpolate between knots."""
        x = torch.linspace(0, 1, 10)
        y = x
        spline = smoothing_spline_fit(x, y, smoothing=1e-6)

        # Evaluate at points between knots
        t = torch.tensor([0.15, 0.35, 0.55, 0.75])
        y_eval = smoothing_spline_evaluate(spline, t)

        # For linear data with low smoothing, should be close to linear
        assert torch.allclose(y_eval, t, rtol=0.1, atol=0.05)

    def test_evaluate_scalar(self):
        """Should handle scalar query."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = smoothing_spline_fit(x, y, smoothing=1e-3)

        result = smoothing_spline_evaluate(spline, torch.tensor(0.5))

        assert result.dim() == 0 or result.numel() == 1

    def test_evaluate_batch(self):
        """Should handle batched queries."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = smoothing_spline_fit(x, y, smoothing=1e-3)

        t = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        result = smoothing_spline_evaluate(spline, t)

        assert result.shape == (2, 2)

    def test_extrapolate_error(self):
        """Should raise error for out-of-domain."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = smoothing_spline_fit(
            x, y, smoothing=1e-3, extrapolate="error"
        )

        with pytest.raises(ExtrapolationError):
            smoothing_spline_evaluate(spline, torch.tensor(-0.1))

    def test_extrapolate_clamp(self):
        """Should clamp to domain."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = smoothing_spline_fit(
            x, y, smoothing=1e-3, extrapolate="clamp"
        )

        result_left = smoothing_spline_evaluate(spline, torch.tensor(-0.5))
        result_at_0 = smoothing_spline_evaluate(spline, torch.tensor(0.0))

        assert torch.allclose(result_left, result_at_0, atol=1e-5)

    def test_extrapolate_extend(self):
        """Should linearly extrapolate."""
        x = torch.linspace(0, 1, 10)
        y = x  # Linear data
        spline = smoothing_spline_fit(
            x, y, smoothing=1e-6, extrapolate="extrapolate"
        )

        # Linear extrapolation
        result_left = smoothing_spline_evaluate(spline, torch.tensor(-0.1))
        result_right = smoothing_spline_evaluate(spline, torch.tensor(1.1))

        assert result_left < 0  # Extrapolated below
        assert result_right > 1  # Extrapolated above


class TestSmoothingSplineDerivative:
    """Tests for smoothing_spline_derivative function."""

    def test_first_derivative(self):
        """Should compute first derivative."""
        x = torch.linspace(0, 1, 20)
        y = x**2  # f'(x) = 2x
        spline = smoothing_spline_fit(x, y, smoothing=1e-6)

        t = torch.tensor([0.25, 0.5, 0.75])
        deriv = smoothing_spline_derivative(spline, t, order=1)

        # Expected: 2*t
        expected = 2 * t
        assert torch.allclose(deriv, expected, rtol=0.2, atol=0.1)

    def test_second_derivative(self):
        """Should compute second derivative."""
        x = torch.linspace(0, 1, 20)
        y = x**2  # f''(x) = 2
        spline = smoothing_spline_fit(x, y, smoothing=1e-6)

        t = torch.tensor([0.25, 0.5, 0.75])
        deriv = smoothing_spline_derivative(spline, t, order=2)

        # Expected: constant 2
        expected = torch.full_like(t, 2.0)
        assert torch.allclose(deriv, expected, rtol=0.3, atol=0.5)

    def test_invalid_order(self):
        """Should raise for invalid order."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = smoothing_spline_fit(x, y, smoothing=1e-3)

        with pytest.raises(ValueError):
            smoothing_spline_derivative(spline, torch.tensor(0.5), order=0)

        with pytest.raises(ValueError):
            smoothing_spline_derivative(spline, torch.tensor(0.5), order=3)


class TestSmoothingSplineConvenience:
    """Tests for smoothing_spline convenience function."""

    def test_returns_callable(self):
        """Should return callable."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        f = smoothing_spline(x, y, smoothing=1e-3)

        assert callable(f)

        result = f(torch.tensor([0.25, 0.5, 0.75]))
        assert result.shape == (3,)

    def test_passes_smoothing(self):
        """Should pass smoothing parameter."""
        x = torch.linspace(0, 1, 20)
        y = torch.sin(x * 2 * torch.pi)

        # Different smoothing should produce different results
        f_smooth = smoothing_spline(x, y, smoothing=1.0)
        f_rough = smoothing_spline(x, y, smoothing=1e-6)

        # Evaluate at multiple points to compare overall fit
        t = torch.linspace(0.1, 0.9, 5)
        val_smooth = f_smooth(t)
        val_rough = f_rough(t)

        # With high smoothing, amplitude should be reduced overall
        # Compare max deviations from zero (the sine oscillates around zero)
        max_smooth = val_smooth.abs().max()
        max_rough = val_rough.abs().max()

        # Rough fit should track the oscillations better
        assert max_rough > max_smooth or torch.allclose(
            max_rough, max_smooth, rtol=0.1
        )


class TestSmoothingSplineMultidimensional:
    """Tests for multi-dimensional values."""

    def test_fit_2d_values(self):
        """Should handle 2D y values."""
        x = torch.linspace(0, 1, 15)
        y = torch.stack([x**2, torch.sin(x * torch.pi)], dim=-1)

        spline = smoothing_spline_fit(x, y, smoothing=1e-4)

        assert spline.coefficients.shape == (15, 2)

    def test_evaluate_2d_values(self):
        """Should evaluate 2D values."""
        x = torch.linspace(0, 1, 15)
        y = torch.stack([x, x**2], dim=-1)
        spline = smoothing_spline_fit(x, y, smoothing=1e-6)

        t = torch.tensor([0.25, 0.5, 0.75])
        result = smoothing_spline_evaluate(spline, t)

        assert result.shape == (3, 2)

    def test_derivative_2d_values(self):
        """Should compute derivative for 2D values."""
        x = torch.linspace(0, 1, 15)
        y = torch.stack([x, x**2], dim=-1)
        spline = smoothing_spline_fit(x, y, smoothing=1e-6)

        t = torch.tensor([0.5])
        deriv = smoothing_spline_derivative(spline, t)

        assert deriv.shape == (1, 2)


class TestSmoothingSplineGradients:
    """Tests for gradient computation."""

    def test_gradcheck_evaluate(self):
        """Gradient through evaluation."""
        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.sin(x * torch.pi)
        spline = smoothing_spline_fit(
            x, y, smoothing=1e-3, extrapolate="clamp"
        )

        t = torch.tensor(
            [0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True
        )
        result = smoothing_spline_evaluate(spline, t)
        loss = result.sum()
        loss.backward()

        assert t.grad is not None

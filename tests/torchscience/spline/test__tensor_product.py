"""Tests for tensor product spline functions."""

import pytest
import torch

from torchscience.spline import (
    ExtrapolationError,
    TensorProductSpline,
    tensor_product_derivative,
    tensor_product_evaluate,
    tensor_product_fit,
    tensor_product_spline,
)


class TestTensorProductFit:
    """Tests for tensor_product_fit function."""

    def test_fit_basic(self):
        """Should fit to 2D gridded data."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 4)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X**2 + Y**2

        spline = tensor_product_fit(x, y, Z)

        assert isinstance(spline, TensorProductSpline)
        assert spline.x_knots.shape == (5,)
        assert spline.y_knots.shape == (4,)
        assert spline.coefficients.shape == (5, 4)

    def test_fit_with_value_shape(self):
        """Should handle multi-dimensional values."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 4)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        # 2D values
        Z = torch.stack([X**2, Y**2], dim=-1)

        spline = tensor_product_fit(x, y, Z)

        assert spline.coefficients.shape == (5, 4, 2)

    def test_fit_requires_2_x_points(self):
        """Should require at least 2 x-coordinates."""
        x = torch.tensor([0.5])
        y = torch.linspace(0, 1, 4)
        Z = torch.zeros(1, 4)

        with pytest.raises(ValueError):
            tensor_product_fit(x, y, Z)

    def test_fit_requires_2_y_points(self):
        """Should require at least 2 y-coordinates."""
        x = torch.linspace(0, 1, 4)
        y = torch.tensor([0.5])
        Z = torch.zeros(4, 1)

        with pytest.raises(ValueError):
            tensor_product_fit(x, y, Z)

    def test_fit_strictly_increasing_x(self):
        """Should require strictly increasing x."""
        x = torch.tensor([0.0, 0.5, 0.5, 1.0])
        y = torch.linspace(0, 1, 4)
        Z = torch.zeros(4, 4)

        with pytest.raises(ValueError):
            tensor_product_fit(x, y, Z)

    def test_fit_strictly_increasing_y(self):
        """Should require strictly increasing y."""
        x = torch.linspace(0, 1, 4)
        y = torch.tensor([0.0, 0.5, 0.5, 1.0])
        Z = torch.zeros(4, 4)

        with pytest.raises(ValueError):
            tensor_product_fit(x, y, Z)

    def test_fit_shape_mismatch(self):
        """Should raise on z shape mismatch."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 4)
        Z = torch.zeros(3, 4)  # Wrong shape

        with pytest.raises(ValueError):
            tensor_product_fit(x, y, Z)


class TestTensorProductEvaluate:
    """Tests for tensor_product_evaluate function."""

    def test_evaluate_at_grid_points(self):
        """Should exactly match at grid points."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 4)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X**2 + Y**2

        spline = tensor_product_fit(x, y, Z)

        # Evaluate at grid points
        qx = X.flatten()
        qy = Y.flatten()
        result = tensor_product_evaluate(spline, qx, qy)

        expected = Z.flatten()
        assert torch.allclose(result, expected, atol=1e-5)

    def test_evaluate_interpolation(self):
        """Should interpolate between grid points."""
        x = torch.linspace(0, 1, 10)
        y = torch.linspace(0, 1, 10)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X + Y  # Linear function

        spline = tensor_product_fit(x, y, Z)

        # Evaluate at midpoints
        qx = torch.tensor([0.15, 0.55, 0.85])
        qy = torch.tensor([0.25, 0.45, 0.75])
        result = tensor_product_evaluate(spline, qx, qy)

        expected = qx + qy
        assert torch.allclose(result, expected, rtol=0.1, atol=0.05)

    def test_evaluate_2d_query(self):
        """Should handle 2D query arrays."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X * Y

        spline = tensor_product_fit(x, y, Z)

        qx = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
        qy = torch.tensor([[0.3, 0.5], [0.7, 0.9]])
        result = tensor_product_evaluate(spline, qx, qy)

        assert result.shape == (2, 2)

    def test_extrapolate_error(self):
        """Should raise error for out-of-domain."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X + Y

        spline = tensor_product_fit(x, y, Z, extrapolate="error")

        with pytest.raises(ExtrapolationError):
            tensor_product_evaluate(
                spline, torch.tensor(-0.1), torch.tensor(0.5)
            )

    def test_extrapolate_clamp(self):
        """Should clamp to domain."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X + Y

        spline = tensor_product_fit(x, y, Z, extrapolate="clamp")

        result_outside = tensor_product_evaluate(
            spline, torch.tensor(-0.5), torch.tensor(0.5)
        )
        result_boundary = tensor_product_evaluate(
            spline, torch.tensor(0.0), torch.tensor(0.5)
        )

        assert torch.allclose(result_outside, result_boundary, atol=1e-5)

    def test_extrapolate_extend(self):
        """Should linearly extrapolate."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X + Y

        spline = tensor_product_fit(x, y, Z, extrapolate="extrapolate")

        # Linear extrapolation
        result = tensor_product_evaluate(
            spline, torch.tensor(1.5), torch.tensor(0.5)
        )

        # Should be approximately 2.0 for x=1.5, y=0.5
        assert result > 1.0


class TestTensorProductDerivative:
    """Tests for tensor_product_derivative function."""

    def test_derivative_dx(self):
        """Should compute partial derivative with respect to x."""
        x = torch.linspace(0, 1, 10)
        y = torch.linspace(0, 1, 10)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = 2 * X + 3 * Y  # df/dx = 2

        spline = tensor_product_fit(x, y, Z)

        qx = torch.tensor([0.5])
        qy = torch.tensor([0.5])
        deriv = tensor_product_derivative(spline, qx, qy, dx=1, dy=0)

        assert torch.allclose(deriv, torch.tensor([2.0]), rtol=0.2, atol=0.1)

    def test_derivative_dy(self):
        """Should compute partial derivative with respect to y."""
        x = torch.linspace(0, 1, 10)
        y = torch.linspace(0, 1, 10)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = 2 * X + 3 * Y  # df/dy = 3

        spline = tensor_product_fit(x, y, Z)

        qx = torch.tensor([0.5])
        qy = torch.tensor([0.5])
        deriv = tensor_product_derivative(spline, qx, qy, dx=0, dy=1)

        assert torch.allclose(deriv, torch.tensor([3.0]), rtol=0.2, atol=0.1)

    def test_derivative_mixed(self):
        """Should compute mixed partial derivative."""
        x = torch.linspace(0, 1, 10)
        y = torch.linspace(0, 1, 10)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X * Y  # d²f/dxdy = 1

        spline = tensor_product_fit(x, y, Z)

        qx = torch.tensor([0.5])
        qy = torch.tensor([0.5])
        deriv = tensor_product_derivative(spline, qx, qy, dx=1, dy=1)

        assert torch.allclose(deriv, torch.tensor([1.0]), rtol=0.2, atol=0.1)

    def test_derivative_zero_order(self):
        """Zero-order derivative should be evaluation."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X + Y

        spline = tensor_product_fit(x, y, Z)

        qx = torch.tensor([0.5])
        qy = torch.tensor([0.5])

        deriv = tensor_product_derivative(spline, qx, qy, dx=0, dy=0)
        val = tensor_product_evaluate(spline, qx, qy)

        assert torch.allclose(deriv, val, atol=1e-5)

    def test_derivative_invalid_order(self):
        """Should raise for negative order."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X + Y

        spline = tensor_product_fit(x, y, Z)

        with pytest.raises(ValueError):
            tensor_product_derivative(
                spline, torch.tensor(0.5), torch.tensor(0.5), dx=-1
            )


class TestTensorProductConvenience:
    """Tests for tensor_product_spline convenience function."""

    def test_returns_callable(self):
        """Should return callable."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X**2 + Y**2

        f = tensor_product_spline(x, y, Z)

        assert callable(f)

        result = f(torch.tensor([0.5]), torch.tensor([0.5]))
        assert result.shape == (1,)

    def test_matches_separate_calls(self):
        """Should match fit + evaluate."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X + Y

        f = tensor_product_spline(x, y, Z)
        spline = tensor_product_fit(x, y, Z)

        qx = torch.tensor([0.25, 0.5, 0.75])
        qy = torch.tensor([0.3, 0.5, 0.7])

        result_convenience = f(qx, qy)
        result_separate = tensor_product_evaluate(spline, qx, qy)

        assert torch.allclose(result_convenience, result_separate, atol=1e-5)


class TestTensorProductGradients:
    """Tests for gradient computation."""

    def test_gradcheck_evaluate(self):
        """Gradient through evaluation."""
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.linspace(0, 1, 5, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X + Y

        spline = tensor_product_fit(x, y, Z, extrapolate="clamp")

        qx = torch.tensor(
            [0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True
        )
        qy = torch.tensor(
            [0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True
        )

        result = tensor_product_evaluate(spline, qx, qy)
        loss = result.sum()
        loss.backward()

        assert qx.grad is not None
        assert qy.grad is not None


class TestTensorProductMultidimensional:
    """Tests for multi-dimensional values."""

    def test_evaluate_2d_values(self):
        """Should handle 2D function values."""
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = torch.stack([X, Y, X + Y], dim=-1)  # (5, 5, 3)

        spline = tensor_product_fit(x, y, Z)

        qx = torch.tensor([0.5])
        qy = torch.tensor([0.5])
        result = tensor_product_evaluate(spline, qx, qy)

        assert result.shape == (1, 3)
        expected = torch.tensor([[0.5, 0.5, 1.0]])
        assert torch.allclose(result, expected, rtol=0.1, atol=0.05)

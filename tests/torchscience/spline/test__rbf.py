"""Tests for RBF interpolation functions."""

import pytest
import torch

from torchscience.spline import (
    RBFInterpolator,
    rbf_evaluate,
    rbf_fit,
    rbf_interpolate,
)


class TestRBFFit:
    """Tests for rbf_fit function."""

    def test_fit_basic_1d(self):
        """Should fit 1D scattered data."""
        points = torch.linspace(0, 1, 10).unsqueeze(-1)
        values = torch.sin(points[:, 0] * torch.pi)

        rbf = rbf_fit(points, values, kernel="thin_plate")

        assert isinstance(rbf, RBFInterpolator)
        assert rbf.centers.shape == (10, 1)
        assert rbf.weights.shape[0] == 10

    def test_fit_basic_2d(self):
        """Should fit 2D scattered data."""
        torch.manual_seed(42)
        points = torch.rand(20, 2)
        values = torch.sin(points[:, 0]) * torch.cos(points[:, 1])

        rbf = rbf_fit(points, values, kernel="thin_plate")

        assert rbf.centers.shape == (20, 2)

    def test_fit_with_epsilon(self):
        """Should accept epsilon for appropriate kernels."""
        points = torch.rand(10, 2)
        values = torch.rand(10)

        rbf = rbf_fit(points, values, kernel="gaussian", epsilon=1.0)

        assert rbf.epsilon == 1.0

    def test_fit_auto_epsilon(self):
        """Should auto-select epsilon when not provided."""
        points = torch.rand(10, 2)
        values = torch.rand(10)

        rbf = rbf_fit(points, values, kernel="gaussian")

        assert rbf.epsilon > 0

    def test_fit_multidimensional_values(self):
        """Should handle multi-dimensional values."""
        points = torch.rand(15, 2)
        values = torch.rand(15, 3)  # 3D values

        rbf = rbf_fit(points, values, kernel="thin_plate")

        assert rbf.weights.shape == (15, 3)


class TestRBFEvaluate:
    """Tests for rbf_evaluate function."""

    def test_evaluate_at_data_points(self):
        """Should interpolate exactly at data points."""
        torch.manual_seed(42)
        points = torch.rand(10, 2, dtype=torch.float64)
        values = torch.sin(points[:, 0]) * torch.cos(points[:, 1])

        rbf = rbf_fit(points, values, kernel="thin_plate")
        result = rbf_evaluate(rbf, points)

        # Should be exact interpolation
        assert torch.allclose(result, values, rtol=1e-4, atol=1e-5)

    def test_evaluate_interpolation(self):
        """Should interpolate smoothly."""
        # 2D grid as scattered points
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        values = points[:, 0] + points[:, 1]  # Linear function

        rbf = rbf_fit(points, values, kernel="thin_plate")

        # Query at interior point
        query = torch.tensor([[0.5, 0.5]])
        result = rbf_evaluate(rbf, query)

        # Should be close to 1.0
        assert torch.allclose(result, torch.tensor([1.0]), rtol=0.1, atol=0.05)

    def test_evaluate_batch_query(self):
        """Should handle batch queries."""
        points = torch.rand(10, 2)
        values = torch.rand(10)

        rbf = rbf_fit(points, values, kernel="gaussian", epsilon=1.0)

        query = torch.rand(5, 2)
        result = rbf_evaluate(rbf, query)

        assert result.shape == (5,)

    def test_evaluate_multidimensional_values(self):
        """Should evaluate multi-dimensional values."""
        points = torch.rand(15, 2)
        values = torch.rand(15, 3)

        rbf = rbf_fit(points, values, kernel="thin_plate")

        query = torch.rand(4, 2)
        result = rbf_evaluate(rbf, query)

        assert result.shape == (4, 3)


class TestRBFKernels:
    """Tests for different RBF kernels."""

    @pytest.mark.parametrize(
        "kernel",
        ["thin_plate", "cubic", "linear"],
    )
    def test_cpd_kernels(self, kernel):
        """CPD kernels should work with polynomial terms."""
        torch.manual_seed(42)
        points = torch.rand(15, 2, dtype=torch.float64)
        values = points[:, 0] + points[:, 1]

        rbf = rbf_fit(points, values, kernel=kernel)
        result = rbf_evaluate(rbf, points)

        assert torch.allclose(result, values, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "kernel",
        [
            "gaussian",
            "multiquadric",
            "inverse_quadratic",
            "inverse_multiquadric",
        ],
    )
    def test_pd_kernels(self, kernel):
        """Positive definite kernels should work without polynomial."""
        torch.manual_seed(42)
        points = torch.rand(15, 2, dtype=torch.float64)
        values = torch.sin(points[:, 0]) * torch.cos(points[:, 1])

        rbf = rbf_fit(points, values, kernel=kernel, epsilon=2.0)
        result = rbf_evaluate(rbf, points)

        assert torch.allclose(result, values, rtol=1e-3, atol=1e-4)

    def test_unknown_kernel(self):
        """Should raise error for unknown kernel."""
        points = torch.rand(10, 2)
        values = torch.rand(10)

        with pytest.raises(ValueError, match="Unknown kernel"):
            rbf_fit(points, values, kernel="unknown")


class TestRBFConvenience:
    """Tests for rbf_interpolate convenience function."""

    def test_returns_callable(self):
        """Should return callable."""
        points = torch.rand(10, 2)
        values = torch.rand(10)

        f = rbf_interpolate(points, values)

        assert callable(f)

        result = f(torch.rand(3, 2))
        assert result.shape == (3,)

    def test_matches_separate_calls(self):
        """Should match fit + evaluate."""
        torch.manual_seed(42)
        points = torch.rand(10, 2)
        values = torch.rand(10)

        f = rbf_interpolate(points, values, kernel="thin_plate")
        rbf = rbf_fit(points, values, kernel="thin_plate")

        query = torch.rand(5, 2)
        result_convenience = f(query)
        result_separate = rbf_evaluate(rbf, query)

        assert torch.allclose(result_convenience, result_separate, atol=1e-5)


class TestRBFSmoothing:
    """Tests for RBF smoothing parameter."""

    def test_smoothing_regularization(self):
        """Smoothing should regularize the fit."""
        torch.manual_seed(42)
        points = torch.rand(20, 2)
        values = torch.sin(points[:, 0]) + 0.3 * torch.randn(20)  # Noisy data

        # No smoothing - should fit exactly
        rbf_exact = rbf_fit(
            points, values, kernel="gaussian", epsilon=2.0, smoothing=0.0
        )
        result_exact = rbf_evaluate(rbf_exact, points)

        # With smoothing - should not fit exactly
        rbf_smooth = rbf_fit(
            points, values, kernel="gaussian", epsilon=2.0, smoothing=0.1
        )
        result_smooth = rbf_evaluate(rbf_smooth, points)

        error_exact = (result_exact - values).abs().mean()
        error_smooth = (result_smooth - values).abs().mean()

        # Exact fit should have smaller error at data points
        assert error_exact < error_smooth


class TestRBFGradients:
    """Tests for gradient computation."""

    def test_gradcheck_evaluate(self):
        """Gradient through evaluation."""
        torch.manual_seed(42)
        points = torch.rand(8, 2, dtype=torch.float64)
        values = torch.sin(points[:, 0]) * torch.cos(points[:, 1])

        rbf = rbf_fit(points, values, kernel="gaussian", epsilon=2.0)

        query = torch.rand(3, 2, dtype=torch.float64, requires_grad=True)
        result = rbf_evaluate(rbf, query)
        loss = result.sum()
        loss.backward()

        assert query.grad is not None


class TestRBF1D:
    """Tests for 1D RBF interpolation."""

    def test_1d_interpolation(self):
        """Should work for 1D data."""
        x = torch.linspace(0, 2 * torch.pi, 10, dtype=torch.float64).unsqueeze(
            -1
        )
        y = torch.sin(x[:, 0])

        rbf = rbf_fit(x, y, kernel="thin_plate")

        # Query at midpoints
        query = torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float64)
        result = rbf_evaluate(rbf, query)

        expected = torch.sin(query[:, 0])
        assert torch.allclose(result, expected, rtol=0.2, atol=0.1)

    def test_1d_query_shape(self):
        """Should handle 1D queries correctly."""
        x = torch.linspace(0, 1, 10).unsqueeze(-1)
        y = x[:, 0] ** 2

        rbf = rbf_fit(x, y, kernel="thin_plate")

        # 1D query
        query = torch.tensor([[0.5]])
        result = rbf_evaluate(rbf, query)

        assert result.shape == (1,)

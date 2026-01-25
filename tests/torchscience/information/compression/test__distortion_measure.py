"""Tests for distortion measures."""

import pytest
import torch

from torchscience.information.compression import distortion_measure


class TestDistortionMeasureBasic:
    """Basic functionality tests."""

    def test_output_type(self):
        """Returns tensor."""
        x = torch.randn(10)
        y = torch.randn(10)
        d = distortion_measure(x, y)
        assert isinstance(d, torch.Tensor)

    def test_output_is_scalar(self):
        """Default reduction gives scalar."""
        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        d = distortion_measure(x, y)
        assert d.dim() == 0

    def test_identical_signals_zero_distortion(self):
        """Identical signals have zero MSE/MAE."""
        x = torch.randn(10)
        mse = distortion_measure(x, x, metric="mse")
        mae = distortion_measure(x, x, metric="mae")
        assert torch.isclose(mse, torch.tensor(0.0))
        assert torch.isclose(mae, torch.tensor(0.0))


class TestDistortionMSE:
    """Tests for mean squared error."""

    def test_mse_formula(self):
        """MSE matches expected formula."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.5, 2.0])
        # MSE = mean((0)^2, (0.5)^2, (1)^2) = (0 + 0.25 + 1) / 3 = 0.4167
        mse = distortion_measure(x, y, metric="mse")
        expected = torch.tensor(0.25 + 1.0).div(3)
        assert torch.isclose(mse, expected)

    def test_mse_non_negative(self):
        """MSE is always non-negative."""
        x = torch.randn(100)
        y = torch.randn(100)
        mse = distortion_measure(x, y, metric="mse")
        assert mse >= 0


class TestDistortionMAE:
    """Tests for mean absolute error."""

    def test_mae_formula(self):
        """MAE matches expected formula."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.5, 2.0])
        # MAE = mean(0, 0.5, 1) = 1.5 / 3 = 0.5
        mae = distortion_measure(x, y, metric="mae")
        expected = torch.tensor(0.5)
        assert torch.isclose(mae, expected)

    def test_mae_non_negative(self):
        """MAE is always non-negative."""
        x = torch.randn(100)
        y = torch.randn(100)
        mae = distortion_measure(x, y, metric="mae")
        assert mae >= 0


class TestDistortionRMSE:
    """Tests for root mean squared error."""

    def test_rmse_is_sqrt_mse(self):
        """RMSE equals sqrt(MSE)."""
        x = torch.randn(50)
        y = torch.randn(50)
        mse = distortion_measure(x, y, metric="mse")
        rmse = distortion_measure(x, y, metric="rmse")
        assert torch.isclose(rmse, mse.sqrt())

    def test_rmse_non_negative(self):
        """RMSE is always non-negative."""
        x = torch.randn(100)
        y = torch.randn(100)
        rmse = distortion_measure(x, y, metric="rmse")
        assert rmse >= 0


class TestDistortionPSNR:
    """Tests for peak signal-to-noise ratio."""

    def test_psnr_identical_is_inf(self):
        """PSNR of identical signals is infinity."""
        x = torch.randn(10)
        psnr = distortion_measure(x, x, metric="psnr")
        assert psnr == float("inf")

    def test_psnr_increases_with_quality(self):
        """Higher quality (less noise) gives higher PSNR."""
        x = torch.randn(100).clamp(0, 1)
        y_good = x + 0.01 * torch.randn(100)  # Low noise
        y_bad = x + 0.1 * torch.randn(100)  # High noise

        psnr_good = distortion_measure(x, y_good, metric="psnr")
        psnr_bad = distortion_measure(x, y_bad, metric="psnr")
        assert psnr_good > psnr_bad

    def test_psnr_reasonable_range(self):
        """PSNR is in reasonable dB range."""
        x = torch.rand(100)  # [0, 1] range
        y = x + 0.05 * torch.randn(100)
        psnr = distortion_measure(x, y, metric="psnr")
        # Typical PSNR for good quality is 30-50 dB
        assert 10 < psnr < 60


class TestDistortionSSIM:
    """Tests for structural similarity approximation."""

    def test_ssim_identical_is_one(self):
        """SSIM of identical signals is 1."""
        x = torch.randn(100)
        ssim = distortion_measure(x, x, metric="ssim_approx")
        assert torch.isclose(ssim, torch.tensor(1.0), atol=1e-5)

    def test_ssim_bounded(self):
        """SSIM is bounded in [-1, 1]."""
        x = torch.randn(100)
        y = torch.randn(100)
        ssim = distortion_measure(x, y, metric="ssim_approx")
        assert -1 <= ssim <= 1

    def test_ssim_higher_for_similar(self):
        """Similar signals have higher SSIM."""
        torch.manual_seed(42)
        x = torch.randn(1000)  # More samples for better statistics
        y_similar = x + 0.01 * torch.randn(1000)  # Very similar
        y_different = -x  # Opposite signal (should have low/negative SSIM)

        ssim_similar = distortion_measure(x, y_similar, metric="ssim_approx")
        ssim_different = distortion_measure(
            x, y_different, metric="ssim_approx"
        )
        assert ssim_similar > ssim_different


class TestDistortionReduction:
    """Tests for reduction modes."""

    def test_reduction_none_preserves_shape(self):
        """None reduction preserves input shape."""
        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        d = distortion_measure(x, y, metric="mse", reduction="none")
        assert d.shape == x.shape

    def test_reduction_sum(self):
        """Sum reduction sums all elements."""
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        d_none = distortion_measure(x, y, metric="mse", reduction="none")
        d_sum = distortion_measure(x, y, metric="mse", reduction="sum")
        assert torch.isclose(d_sum, d_none.sum())

    def test_reduction_mean(self):
        """Mean reduction averages all elements."""
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        d_none = distortion_measure(x, y, metric="mse", reduction="none")
        d_mean = distortion_measure(x, y, metric="mse", reduction="mean")
        assert torch.isclose(d_mean, d_none.mean())


class TestDistortionGradients:
    """Tests for gradient computation."""

    def test_mse_gradient(self):
        """MSE is differentiable."""
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10)
        d = distortion_measure(x, y, metric="mse")
        d.backward()
        assert x.grad is not None

    def test_mae_gradient(self):
        """MAE is differentiable."""
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10)
        d = distortion_measure(x, y, metric="mae")
        d.backward()
        assert x.grad is not None

    def test_ssim_gradient(self):
        """SSIM approximation is differentiable."""
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10)
        d = distortion_measure(x, y, metric="ssim_approx")
        d.backward()
        assert x.grad is not None


class TestDistortionEdgeCases:
    """Edge case tests."""

    def test_single_element(self):
        """Works with single element tensors."""
        x = torch.tensor([1.0])
        y = torch.tensor([1.5])
        d = distortion_measure(x, y, metric="mse")
        assert torch.isclose(d, torch.tensor(0.25))

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        y = torch.randn(10)
        with pytest.raises(TypeError, match="must be a Tensor"):
            distortion_measure([1.0], y)

    def test_shape_mismatch_raises(self):
        """Raises error for shape mismatch."""
        x = torch.randn(10)
        y = torch.randn(20)
        with pytest.raises(ValueError, match="shape"):
            distortion_measure(x, y)

    def test_invalid_metric_raises(self):
        """Raises error for invalid metric."""
        x = torch.randn(10)
        y = torch.randn(10)
        with pytest.raises(ValueError, match="metric"):
            distortion_measure(x, y, metric="invalid")

    def test_invalid_reduction_raises(self):
        """Raises error for invalid reduction."""
        x = torch.randn(10)
        y = torch.randn(10)
        with pytest.raises(ValueError, match="reduction"):
            distortion_measure(x, y, reduction="invalid")

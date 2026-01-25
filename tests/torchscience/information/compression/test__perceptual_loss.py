"""Tests for perceptual loss functions."""

import pytest
import torch

from torchscience.information.compression._perceptual_loss import (
    perceptual_loss,
    rate_loss,
)


class TestPerceptualLoss:
    """Tests for perceptual_loss function."""

    def test_ssim_output_scalar(self):
        """SSIM loss returns scalar with mean reduction."""
        x = torch.randn(4, 3, 64, 64)
        y = torch.randn(4, 3, 64, 64)
        loss = perceptual_loss(x, y, method="ssim")
        assert loss.dim() == 0

    def test_ssim_identical_images(self):
        """SSIM loss is 0 for identical images."""
        x = torch.randn(2, 3, 32, 32)
        loss = perceptual_loss(x, x, method="ssim")
        assert loss.item() < 1e-5

    def test_ssim_different_images(self):
        """SSIM loss is positive for different images."""
        x = torch.randn(2, 3, 32, 32)
        y = torch.randn(2, 3, 32, 32)
        loss = perceptual_loss(x, y, method="ssim")
        assert loss.item() > 0

    def test_ssim_bounded(self):
        """SSIM loss is bounded in [0, 2]."""
        x = torch.randn(2, 3, 32, 32)
        y = torch.randn(2, 3, 32, 32)
        loss = perceptual_loss(x, y, method="ssim")
        assert 0 <= loss.item() <= 2

    def test_ms_ssim_output_scalar(self):
        """MS-SSIM loss returns scalar."""
        x = torch.randn(2, 3, 64, 64)
        y = torch.randn(2, 3, 64, 64)
        loss = perceptual_loss(x, y, method="ms_ssim")
        assert loss.dim() == 0

    def test_ms_ssim_identical_images(self):
        """MS-SSIM loss is near 0 for identical images."""
        x = torch.randn(2, 3, 64, 64)
        loss = perceptual_loss(x, x, method="ms_ssim")
        assert loss.item() < 0.01

    def test_gradient_method(self):
        """Gradient loss works."""
        x = torch.randn(2, 3, 32, 32)
        y = torch.randn(2, 3, 32, 32)
        loss = perceptual_loss(x, y, method="gradient")
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_identical_images(self):
        """Gradient loss is 0 for identical images."""
        x = torch.randn(2, 1, 32, 32)
        loss = perceptual_loss(x, x, method="gradient")
        assert loss.item() < 1e-5

    def test_laplacian_method(self):
        """Laplacian loss works."""
        x = torch.randn(2, 3, 64, 64)
        y = torch.randn(2, 3, 64, 64)
        loss = perceptual_loss(x, y, method="laplacian")
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_laplacian_identical_images(self):
        """Laplacian loss is 0 for identical images."""
        x = torch.randn(2, 3, 64, 64)
        loss = perceptual_loss(x, x, method="laplacian")
        assert loss.item() < 1e-5

    def test_reduction_none(self):
        """Reduction none returns per-batch loss."""
        x = torch.randn(4, 3, 32, 32)
        y = torch.randn(4, 3, 32, 32)
        loss = perceptual_loss(x, y, reduction="none")
        assert loss.shape == (4,)

    def test_reduction_sum(self):
        """Reduction sum returns summed loss."""
        x = torch.randn(4, 3, 32, 32)
        y = torch.randn(4, 3, 32, 32)
        loss_sum = perceptual_loss(x, y, reduction="sum")
        loss_none = perceptual_loss(x, y, reduction="none")
        assert torch.allclose(loss_sum, loss_none.sum())

    def test_gradients_flow(self):
        """Gradients flow through the loss."""
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        y = torch.randn(2, 3, 32, 32)
        loss = perceptual_loss(x, y, method="ssim")
        loss.backward()
        assert x.grad is not None

    def test_single_channel(self):
        """Works with single channel images."""
        x = torch.randn(2, 1, 32, 32)
        y = torch.randn(2, 1, 32, 32)
        loss = perceptual_loss(x, y, method="ssim")
        assert loss.dim() == 0

    def test_not_tensor_x_raises(self):
        """Raises error for non-tensor x."""
        y = torch.randn(2, 3, 32, 32)
        with pytest.raises(TypeError, match="x must be a Tensor"):
            perceptual_loss([[[1.0]]], y)

    def test_not_tensor_y_raises(self):
        """Raises error for non-tensor y."""
        x = torch.randn(2, 3, 32, 32)
        with pytest.raises(TypeError, match="y must be a Tensor"):
            perceptual_loss(x, [[[1.0]]])

    def test_shape_mismatch_raises(self):
        """Raises error for shape mismatch."""
        x = torch.randn(2, 3, 32, 32)
        y = torch.randn(2, 3, 64, 64)
        with pytest.raises(ValueError, match="must match"):
            perceptual_loss(x, y)

    def test_wrong_dims_raises(self):
        """Raises error for non-4D input."""
        x = torch.randn(3, 32, 32)
        y = torch.randn(3, 32, 32)
        with pytest.raises(ValueError, match="4D"):
            perceptual_loss(x, y)

    def test_invalid_method_raises(self):
        """Raises error for invalid method."""
        x = torch.randn(2, 3, 32, 32)
        y = torch.randn(2, 3, 32, 32)
        with pytest.raises(ValueError, match="method must be"):
            perceptual_loss(x, y, method="invalid")

    def test_invalid_reduction_raises(self):
        """Raises error for invalid reduction."""
        x = torch.randn(2, 3, 32, 32)
        y = torch.randn(2, 3, 32, 32)
        with pytest.raises(ValueError, match="reduction must be"):
            perceptual_loss(x, y, reduction="invalid")


class TestRateLoss:
    """Tests for rate_loss function."""

    def test_single_tensor_output_scalar(self):
        """Single tensor input returns scalar."""
        likelihoods = torch.rand(4, 32, 8, 8) * 0.9 + 0.1
        rate = rate_loss(likelihoods)
        assert rate.dim() == 0

    def test_list_of_tensors(self):
        """List of tensors works."""
        lik1 = torch.rand(4, 32, 8, 8) * 0.9 + 0.1
        lik2 = torch.rand(4, 64, 4, 4) * 0.9 + 0.1
        rate = rate_loss([lik1, lik2])
        assert rate.dim() == 0

    def test_rate_positive(self):
        """Rate is always non-negative."""
        likelihoods = torch.rand(4, 32, 8, 8) * 0.9 + 0.1
        rate = rate_loss(likelihoods)
        assert rate.item() >= 0

    def test_higher_likelihood_lower_rate(self):
        """Higher likelihood gives lower rate."""
        lik_high = torch.ones(4, 32, 8, 8) * 0.9
        lik_low = torch.ones(4, 32, 8, 8) * 0.1
        rate_high = rate_loss(lik_high)
        rate_low = rate_loss(lik_low)
        assert rate_high < rate_low

    def test_reduction_none(self):
        """Reduction none returns per-batch rate."""
        likelihoods = torch.rand(4, 32, 8, 8) * 0.9 + 0.1
        rate = rate_loss(likelihoods, reduction="none")
        assert rate.shape == (4,)

    def test_reduction_sum(self):
        """Reduction sum returns total rate."""
        likelihoods = torch.rand(4, 32, 8, 8) * 0.9 + 0.1
        rate_sum = rate_loss(likelihoods, reduction="sum")
        rate_none = rate_loss(likelihoods, reduction="none")
        assert torch.allclose(rate_sum, rate_none.sum())

    def test_gradients_flow(self):
        """Gradients flow through rate loss."""
        likelihoods = torch.rand(2, 32, 8, 8) * 0.9 + 0.1
        likelihoods.requires_grad_(True)
        rate = rate_loss(likelihoods)
        rate.backward()
        assert likelihoods.grad is not None

    def test_known_value(self):
        """Check against known value."""
        # Likelihood of 0.5 means 1 bit per element
        likelihoods = torch.ones(1, 1, 1, 1) * 0.5
        rate = rate_loss(likelihoods, reduction="sum")
        assert torch.allclose(rate, torch.tensor(1.0), atol=1e-5)

    def test_invalid_reduction_raises(self):
        """Raises error for invalid reduction."""
        likelihoods = torch.rand(4, 32, 8, 8)
        with pytest.raises(ValueError, match="reduction must be"):
            rate_loss(likelihoods, reduction="invalid")


class TestPerceptualLossDevice:
    """Device compatibility tests."""

    def test_perceptual_loss_cpu(self):
        """perceptual_loss works on CPU."""
        x = torch.randn(2, 3, 32, 32, device="cpu")
        y = torch.randn(2, 3, 32, 32, device="cpu")
        loss = perceptual_loss(x, y)
        assert loss.device.type == "cpu"

    def test_rate_loss_cpu(self):
        """rate_loss works on CPU."""
        lik = torch.rand(2, 32, 8, 8, device="cpu") * 0.9 + 0.1
        rate = rate_loss(lik)
        assert rate.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_perceptual_loss_cuda(self):
        """perceptual_loss works on CUDA."""
        x = torch.randn(2, 3, 32, 32, device="cuda")
        y = torch.randn(2, 3, 32, 32, device="cuda")
        loss = perceptual_loss(x, y)
        assert loss.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_rate_loss_cuda(self):
        """rate_loss works on CUDA."""
        lik = torch.rand(2, 32, 8, 8, device="cuda") * 0.9 + 0.1
        rate = rate_loss(lik)
        assert rate.device.type == "cuda"

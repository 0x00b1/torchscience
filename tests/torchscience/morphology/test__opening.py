"""Tests for morphological opening."""

import torch
from torch.autograd import gradcheck

from torchscience.morphology import opening


class TestOpeningShape:
    """Tests for shape handling."""

    def test_2d_image(self):
        """2D image opening."""
        image = torch.rand(64, 64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = opening(image, se)
        assert result.shape == (64, 64)

    def test_2d_batch(self):
        """Batch of 2D images."""
        batch = torch.rand(8, 64, 64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = opening(batch, se)
        assert result.shape == (8, 64, 64)


class TestOpeningProperties:
    """Tests for mathematical properties of opening."""

    def test_anti_extensive(self):
        """Opening is anti-extensive: opening(f) <= f."""
        image = torch.rand(16, 16)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = opening(image, se, padding_mode="replicate")
        assert (result <= image + 1e-6).all()

    def test_idempotent(self):
        """Opening is idempotent: opening(opening(f)) = opening(f)."""
        image = torch.rand(16, 16)
        se = torch.ones(3, 3, dtype=torch.bool)
        once = opening(image, se, padding_mode="replicate")
        twice = opening(once, se, padding_mode="replicate")
        assert torch.allclose(once, twice, atol=1e-5)

    def test_removes_small_bright_spots(self):
        """Opening removes small bright features."""
        image = torch.zeros(20, 20)
        image[10, 10] = 1.0  # Single bright pixel
        se = torch.ones(3, 3, dtype=torch.bool)
        result = opening(image, se, padding_mode="replicate")
        # The single pixel should be removed
        assert result[10, 10].item() == 0.0

    def test_preserves_large_structures(self):
        """Opening approximately preserves larger structures."""
        image = torch.zeros(20, 20)
        image[5:15, 5:15] = 1.0  # 10x10 bright square
        se = torch.ones(3, 3, dtype=torch.bool)
        result = opening(image, se, padding_mode="replicate")
        # Most of the square should remain
        assert result[10, 10].item() == 1.0


class TestOpeningGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Gradient check for opening."""
        image = torch.rand(8, 8, dtype=torch.float64, requires_grad=True)
        se = torch.ones(3, 3, dtype=torch.bool)
        assert gradcheck(
            lambda x: opening(x, se, padding_mode="replicate"),
            (image,),
            eps=1e-6,
            atol=1e-4,
            nondet_tol=1e-4,
        )

    def test_gradient_flows(self):
        """Verify gradients flow back correctly."""
        image = torch.rand(16, 16, requires_grad=True)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = opening(image, se)
        loss = result.sum()
        loss.backward()
        assert image.grad is not None
        assert not torch.isnan(image.grad).any()


class TestOpeningDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        image = torch.rand(16, 16, dtype=torch.float32)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = opening(image, se)
        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        image = torch.rand(16, 16, dtype=torch.float64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = opening(image, se)
        assert result.dtype == torch.float64

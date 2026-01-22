"""Tests for morphological erosion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.morphology import erosion


class TestErosionShape:
    """Tests for shape handling."""

    def test_2d_image(self):
        """2D image erosion."""
        image = torch.rand(64, 64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(image, se)
        assert result.shape == (64, 64)

    def test_2d_batch(self):
        """Batch of 2D images."""
        batch = torch.rand(8, 64, 64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(batch, se)
        assert result.shape == (8, 64, 64)

    def test_3d_volume(self):
        """3D volume erosion."""
        volume = torch.rand(32, 32, 32)
        se = torch.ones(3, 3, 3, dtype=torch.bool)
        result = erosion(volume, se)
        assert result.shape == (32, 32, 32)

    def test_1d_signal(self):
        """1D signal erosion."""
        signal = torch.rand(100)
        se = torch.ones(5, dtype=torch.bool)
        result = erosion(signal, se)
        assert result.shape == (100,)

    def test_channel_batch(self):
        """Batch with channel dimension (B, C, H, W)."""
        images = torch.rand(4, 3, 32, 32)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(images, se)
        assert result.shape == (4, 3, 32, 32)


class TestErosionKnownValues:
    """Tests for known erosion values."""

    def test_constant_image(self):
        """Erosion of constant image equals the constant."""
        image = torch.full((10, 10), 5.0)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(image, se, padding_mode="replicate")
        assert torch.allclose(result, image)

    def test_binary_erosion_shrinks(self):
        """Binary erosion shrinks a square."""
        image = torch.zeros(10, 10)
        image[3:7, 3:7] = 1.0  # 4x4 square
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(image, se, padding_mode="replicate")
        # The square should shrink by 1 pixel on each side
        expected = torch.zeros(10, 10)
        expected[4:6, 4:6] = 1.0  # 2x2 square
        assert torch.allclose(result, expected)

    def test_minimum_over_neighborhood(self):
        """Erosion computes local minimum."""
        image = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(image, se, padding_mode="replicate")
        # Center pixel should be minimum of entire 3x3 = 1.0
        assert result[1, 1].item() == 1.0


class TestErosionPaddingModes:
    """Tests for different padding modes."""

    def test_zeros_padding(self):
        """Zero padding uses +inf for erosion."""
        image = torch.ones(5, 5)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(image, se, padding_mode="zeros")
        # Interior should remain 1.0
        assert result[2, 2].item() == 1.0
        # But border should be inf (from padding)
        assert result[0, 0].item() == 1.0  # Corners touch only padding=+inf

    def test_replicate_padding(self):
        """Replicate padding extends edge values."""
        image = torch.ones(5, 5) * 2.0
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(image, se, padding_mode="replicate")
        # All values should remain 2.0
        assert torch.allclose(result, image)


class TestErosionGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Gradient check for erosion."""
        image = torch.rand(8, 8, dtype=torch.float64, requires_grad=True)
        se = torch.ones(3, 3, dtype=torch.bool)
        assert gradcheck(
            lambda x: erosion(x, se, padding_mode="replicate"),
            (image,),
            eps=1e-6,
            atol=1e-4,
            nondet_tol=1e-4,
        )

    def test_gradient_flows(self):
        """Verify gradients flow back correctly."""
        image = torch.rand(16, 16, requires_grad=True)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(image, se)
        loss = result.sum()
        loss.backward()
        assert image.grad is not None
        assert not torch.isnan(image.grad).any()

    def test_gradient_sparsity(self):
        """Gradient should be sparse (only at argmin positions)."""
        image = torch.rand(8, 8, requires_grad=True)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(image, se, padding_mode="replicate")
        loss = result.sum()
        loss.backward()
        # Gradient should have many zeros (most positions aren't argmin)
        num_nonzero = (image.grad != 0).sum().item()
        # At most one gradient per output position
        assert num_nonzero <= result.numel()


class TestErosionDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        image = torch.rand(16, 16, dtype=torch.float32)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(image, se)
        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        image = torch.rand(16, 16, dtype=torch.float64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = erosion(image, se)
        assert result.dtype == torch.float64


class TestErosionGrayscale:
    """Tests for grayscale (non-flat) erosion."""

    def test_grayscale_se(self):
        """Grayscale structuring element subtracts weights."""
        image = torch.ones(5, 5) * 10.0
        se = torch.tensor([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
        result = erosion(image, se, padding_mode="replicate")
        # min(10 - 0, 10 - 1, 10 - 2, ...) = 10 - 2 = 8
        assert torch.isclose(result[2, 2], torch.tensor(8.0))


class TestErosionOrigin:
    """Tests for custom origin."""

    def test_custom_origin(self):
        """Custom origin shifts the SE anchor point."""
        image = torch.rand(10, 10)
        se = torch.ones(3, 3, dtype=torch.bool)
        # Default origin is center (1, 1)
        result_default = erosion(image, se, padding_mode="replicate")
        # Origin at (0, 0)
        result_custom = erosion(
            image, se, origin=[0, 0], padding_mode="replicate"
        )
        # Results should differ
        assert not torch.allclose(result_default, result_custom)


class TestErosionErrors:
    """Tests for error handling."""

    def test_invalid_padding_mode(self):
        """Raise error for invalid padding mode."""
        image = torch.rand(10, 10)
        se = torch.ones(3, 3, dtype=torch.bool)
        with pytest.raises(ValueError, match="padding_mode"):
            erosion(image, se, padding_mode="invalid")

    def test_se_larger_than_input(self):
        """Raise error when SE has more dimensions than input."""
        image = torch.rand(10)  # 1D
        se = torch.ones(3, 3, dtype=torch.bool)  # 2D
        with pytest.raises(ValueError, match="dimensions"):
            erosion(image, se)

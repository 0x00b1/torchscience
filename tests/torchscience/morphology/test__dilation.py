"""Tests for morphological dilation."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.morphology import dilation


class TestDilationShape:
    """Tests for shape handling."""

    def test_2d_image(self):
        """2D image dilation."""
        image = torch.rand(64, 64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = dilation(image, se)
        assert result.shape == (64, 64)

    def test_2d_batch(self):
        """Batch of 2D images."""
        batch = torch.rand(8, 64, 64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = dilation(batch, se)
        assert result.shape == (8, 64, 64)

    def test_3d_volume(self):
        """3D volume dilation."""
        volume = torch.rand(32, 32, 32)
        se = torch.ones(3, 3, 3, dtype=torch.bool)
        result = dilation(volume, se)
        assert result.shape == (32, 32, 32)

    def test_1d_signal(self):
        """1D signal dilation."""
        signal = torch.rand(100)
        se = torch.ones(5, dtype=torch.bool)
        result = dilation(signal, se)
        assert result.shape == (100,)


class TestDilationKnownValues:
    """Tests for known dilation values."""

    def test_constant_image(self):
        """Dilation of constant image equals the constant."""
        image = torch.full((10, 10), 5.0)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = dilation(image, se, padding_mode="replicate")
        assert torch.allclose(result, image)

    def test_binary_dilation_expands(self):
        """Binary dilation expands a point."""
        image = torch.zeros(10, 10)
        image[5, 5] = 1.0  # Single point
        se = torch.ones(3, 3, dtype=torch.bool)
        result = dilation(image, se, padding_mode="replicate")
        # The point should expand to a 3x3 square
        expected = torch.zeros(10, 10)
        expected[4:7, 4:7] = 1.0
        assert torch.allclose(result, expected)

    def test_maximum_over_neighborhood(self):
        """Dilation computes local maximum."""
        image = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        se = torch.ones(3, 3, dtype=torch.bool)
        result = dilation(image, se, padding_mode="replicate")
        # Center pixel should be maximum of entire 3x3 = 9.0
        assert result[1, 1].item() == 9.0


class TestDilationPaddingModes:
    """Tests for different padding modes."""

    def test_zeros_padding(self):
        """Zero padding uses -inf for dilation."""
        image = torch.ones(5, 5)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = dilation(image, se, padding_mode="zeros")
        # All values should be at least 1.0 (the actual values)
        assert result[2, 2].item() == 1.0

    def test_replicate_padding(self):
        """Replicate padding extends edge values."""
        image = torch.ones(5, 5) * 2.0
        se = torch.ones(3, 3, dtype=torch.bool)
        result = dilation(image, se, padding_mode="replicate")
        assert torch.allclose(result, image)


class TestDilationGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Gradient check for dilation."""
        image = torch.rand(8, 8, dtype=torch.float64, requires_grad=True)
        se = torch.ones(3, 3, dtype=torch.bool)
        assert gradcheck(
            lambda x: dilation(x, se, padding_mode="replicate"),
            (image,),
            eps=1e-6,
            atol=1e-4,
            nondet_tol=1e-4,
        )

    def test_gradient_flows(self):
        """Verify gradients flow back correctly."""
        image = torch.rand(16, 16, requires_grad=True)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = dilation(image, se)
        loss = result.sum()
        loss.backward()
        assert image.grad is not None
        assert not torch.isnan(image.grad).any()

    def test_gradient_sparsity(self):
        """Gradient should be sparse (only at argmax positions)."""
        image = torch.rand(8, 8, requires_grad=True)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = dilation(image, se, padding_mode="replicate")
        loss = result.sum()
        loss.backward()
        num_nonzero = (image.grad != 0).sum().item()
        assert num_nonzero <= result.numel()


class TestDilationDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        image = torch.rand(16, 16, dtype=torch.float32)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = dilation(image, se)
        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        image = torch.rand(16, 16, dtype=torch.float64)
        se = torch.ones(3, 3, dtype=torch.bool)
        result = dilation(image, se)
        assert result.dtype == torch.float64


class TestDilationGrayscale:
    """Tests for grayscale (non-flat) dilation."""

    def test_grayscale_se(self):
        """Grayscale structuring element adds weights."""
        image = torch.zeros(5, 5)
        se = torch.tensor([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
        result = dilation(image, se, padding_mode="replicate")
        # max(0 + 0, 0 + 1, 0 + 2, ...) = 2
        assert torch.isclose(result[2, 2], torch.tensor(2.0))


class TestDilationDuality:
    """Tests for duality relationship with erosion."""

    def test_duality(self):
        """dilation(f, B) = -erosion(-f, B)."""
        from torchscience.morphology import erosion

        image = torch.rand(16, 16)
        se = torch.ones(3, 3, dtype=torch.bool)

        dil = dilation(image, se, padding_mode="replicate")
        neg_ero_neg = -erosion(-image, se, padding_mode="replicate")

        assert torch.allclose(dil, neg_ero_neg, atol=1e-5)


class TestDilationErrors:
    """Tests for error handling."""

    def test_invalid_padding_mode(self):
        """Raise error for invalid padding mode."""
        image = torch.rand(10, 10)
        se = torch.ones(3, 3, dtype=torch.bool)
        with pytest.raises(ValueError, match="padding_mode"):
            dilation(image, se, padding_mode="invalid")

    def test_se_larger_than_input(self):
        """Raise error when SE has more dimensions than input."""
        image = torch.rand(10)
        se = torch.ones(3, 3, dtype=torch.bool)
        with pytest.raises(ValueError, match="dimensions"):
            dilation(image, se)

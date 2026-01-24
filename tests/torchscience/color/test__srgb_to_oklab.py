"""Tests for srgb_to_oklab color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_oklab


class TestSrgbToOklabShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        oklab = srgb_to_oklab(rgb)
        assert oklab.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        rgb = torch.randn(10, 3).abs()
        oklab = srgb_to_oklab(rgb)
        assert oklab.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        rgb = torch.randn(64, 64, 3).abs()
        oklab = srgb_to_oklab(rgb)
        assert oklab.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        rgb = torch.randn(10, 32, 32, 3).abs()
        oklab = srgb_to_oklab(rgb)
        assert oklab.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            srgb_to_oklab(torch.randn(10, 4))


class TestSrgbToOklabKnownValues:
    """Tests for known color conversions (reference: Oklab paper)."""

    def test_white(self):
        """White: (1, 1, 1) -> (1, 0, 0)."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        oklab = srgb_to_oklab(rgb)
        # White in Oklab should be L=1, a=0, b=0
        assert torch.isclose(oklab[0], torch.tensor(1.0), atol=1e-4)
        assert torch.isclose(oklab[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(oklab[2], torch.tensor(0.0), atol=1e-4)

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0, 0)."""
        rgb = torch.tensor([0.0, 0.0, 0.0])
        oklab = srgb_to_oklab(rgb)
        # Black should have L=0
        assert torch.isclose(oklab[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(oklab[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(oklab[2], torch.tensor(0.0), atol=1e-4)

    def test_red(self):
        """Pure red sRGB primary."""
        rgb = torch.tensor([1.0, 0.0, 0.0])
        oklab = srgb_to_oklab(rgb)
        # Reference values from Oklab implementation
        # L ~ 0.628, a ~ 0.225, b ~ 0.126
        assert torch.isclose(oklab[0], torch.tensor(0.6280), atol=0.01)
        assert oklab[1] > 0  # positive a (red)
        assert oklab[2] > 0  # positive b (yellow-ish for red)

    def test_green(self):
        """Pure green sRGB primary."""
        rgb = torch.tensor([0.0, 1.0, 0.0])
        oklab = srgb_to_oklab(rgb)
        # Reference values from Oklab implementation
        # L ~ 0.866, a ~ -0.234, b ~ 0.179
        assert torch.isclose(oklab[0], torch.tensor(0.8664), atol=0.01)
        assert oklab[1] < 0  # negative a (green)
        assert oklab[2] > 0  # positive b (yellow-ish for green)

    def test_blue(self):
        """Pure blue sRGB primary."""
        rgb = torch.tensor([0.0, 0.0, 1.0])
        oklab = srgb_to_oklab(rgb)
        # Reference values from Oklab implementation
        # L ~ 0.452, a ~ -0.032, b ~ -0.312
        assert torch.isclose(oklab[0], torch.tensor(0.4520), atol=0.01)
        assert oklab[2] < 0  # negative b (blue)

    def test_mid_gray(self):
        """Mid gray (0.5, 0.5, 0.5) - should have a=0, b=0."""
        rgb = torch.tensor([0.5, 0.5, 0.5])
        oklab = srgb_to_oklab(rgb)
        # Gray should have a=0, b=0
        assert torch.isclose(oklab[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(oklab[2], torch.tensor(0.0), atol=1e-4)
        # L should be between 0 and 1
        assert oklab[0] > 0 and oklab[0] < 1


class TestSrgbToOklabGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        rgb = (
            torch.rand(5, 3, dtype=torch.float64, requires_grad=True) * 0.9
            + 0.05
        )
        assert gradcheck(srgb_to_oklab, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        rgb = (
            torch.rand(4, 4, 3, dtype=torch.float64, requires_grad=True) * 0.9
            + 0.05
        )
        assert gradcheck(srgb_to_oklab, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        rgb = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)
        oklab = srgb_to_oklab(rgb)
        loss = oklab.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.any(torch.isnan(rgb.grad))


class TestSrgbToOklabDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        rgb = torch.rand(10, 3, dtype=torch.float32)
        oklab = srgb_to_oklab(rgb)
        assert oklab.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        rgb = torch.rand(10, 3, dtype=torch.float64)
        oklab = srgb_to_oklab(rgb)
        assert oklab.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        rgb = torch.rand(10, 3, dtype=torch.bfloat16)
        oklab = srgb_to_oklab(rgb)
        assert oklab.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        rgb = torch.rand(10, 3, dtype=torch.float16)
        oklab = srgb_to_oklab(rgb)
        assert oklab.dtype == torch.float16


class TestSrgbToOklabEdgeCases:
    """Tests for edge cases."""

    def test_hdr_values(self):
        """Test with HDR values > 1."""
        rgb = torch.tensor([1.5, 1.2, 0.8])
        oklab = srgb_to_oklab(rgb)
        assert not torch.any(torch.isnan(oklab))
        assert oklab[0] > 1  # L should exceed 1 for HDR

    def test_negative_values(self):
        """Test with negative values (out of gamut)."""
        rgb = torch.tensor([-0.1, 0.5, 0.5])
        oklab = srgb_to_oklab(rgb)
        assert not torch.any(torch.isnan(oklab))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        rgb = torch.empty(0, 3)
        oklab = srgb_to_oklab(rgb)
        assert oklab.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        rgb = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not rgb.is_contiguous()
        oklab = srgb_to_oklab(rgb)
        assert oklab.shape == rgb.shape


class TestSrgbToOklabPerceptualUniformity:
    """Tests for perceptual uniformity properties."""

    def test_lightness_ordering(self):
        """Lighter colors should have higher L values."""
        dark = torch.tensor([0.2, 0.2, 0.2])
        light = torch.tensor([0.8, 0.8, 0.8])
        oklab_dark = srgb_to_oklab(dark)
        oklab_light = srgb_to_oklab(light)
        assert oklab_light[0] > oklab_dark[0]

    def test_chromatic_channels_neutral_for_grays(self):
        """Gray colors should have a=0, b=0."""
        grays = torch.tensor(
            [
                [0.1, 0.1, 0.1],
                [0.3, 0.3, 0.3],
                [0.5, 0.5, 0.5],
                [0.7, 0.7, 0.7],
                [0.9, 0.9, 0.9],
            ]
        )
        oklab = srgb_to_oklab(grays)
        assert torch.allclose(oklab[:, 1], torch.zeros(5), atol=1e-4)
        assert torch.allclose(oklab[:, 2], torch.zeros(5), atol=1e-4)

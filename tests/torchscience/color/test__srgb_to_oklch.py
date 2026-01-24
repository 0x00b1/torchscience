"""Tests for srgb_to_oklch color conversion."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_oklch


class TestSrgbToOklchShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        lch = srgb_to_oklch(rgb)
        assert lch.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        rgb = torch.randn(10, 3).abs()
        lch = srgb_to_oklch(rgb)
        assert lch.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        rgb = torch.randn(64, 64, 3).abs()
        lch = srgb_to_oklch(rgb)
        assert lch.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        rgb = torch.randn(10, 32, 32, 3).abs()
        lch = srgb_to_oklch(rgb)
        assert lch.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            srgb_to_oklch(torch.randn(10, 4))


class TestSrgbToOklchKnownValues:
    """Tests for known color conversions."""

    def test_white(self):
        """White: (1, 1, 1) -> L=1, C=0, h=0."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        lch = srgb_to_oklch(rgb)
        # White in Oklch should be L=1, C=0, h=0 (hue undefined when C=0)
        assert torch.isclose(lch[0], torch.tensor(1.0), atol=1e-4)
        assert torch.isclose(lch[1], torch.tensor(0.0), atol=1e-4)
        # h is undefined when C=0, but we set it to 0

    def test_black(self):
        """Black: (0, 0, 0) -> L=0, C=0."""
        rgb = torch.tensor([0.0, 0.0, 0.0])
        lch = srgb_to_oklch(rgb)
        # Black should have L=0, C=0
        assert torch.isclose(lch[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(lch[1], torch.tensor(0.0), atol=1e-4)

    def test_red(self):
        """Pure red sRGB primary."""
        rgb = torch.tensor([1.0, 0.0, 0.0])
        lch = srgb_to_oklch(rgb)
        # Reference: L ~ 0.628, a ~ 0.2249, b ~ 0.1262
        # C = sqrt(a^2 + b^2) ~ 0.258
        assert torch.isclose(lch[0], torch.tensor(0.6280), atol=0.01)
        expected_c = math.sqrt(0.2249**2 + 0.1262**2)
        assert torch.isclose(lch[1], torch.tensor(expected_c), atol=0.01)
        # h = atan2(b, a)
        expected_h = math.atan2(0.1262, 0.2249)
        assert torch.isclose(lch[2], torch.tensor(expected_h), atol=0.02)

    def test_green(self):
        """Pure green sRGB primary."""
        rgb = torch.tensor([0.0, 1.0, 0.0])
        lch = srgb_to_oklch(rgb)
        # L should be around 0.866
        assert torch.isclose(lch[0], torch.tensor(0.8664), atol=0.01)
        # C should be positive
        assert lch[1] > 0
        # h should be in the green region

    def test_blue(self):
        """Pure blue sRGB primary."""
        rgb = torch.tensor([0.0, 0.0, 1.0])
        lch = srgb_to_oklch(rgb)
        # L should be around 0.452
        assert torch.isclose(lch[0], torch.tensor(0.4520), atol=0.01)
        # C should be positive
        assert lch[1] > 0

    def test_mid_gray(self):
        """Mid gray (0.5, 0.5, 0.5) - should have C=0."""
        rgb = torch.tensor([0.5, 0.5, 0.5])
        lch = srgb_to_oklch(rgb)
        # Gray should have C=0
        assert torch.isclose(lch[1], torch.tensor(0.0), atol=1e-4)
        # L should be between 0 and 1
        assert lch[0] > 0 and lch[0] < 1

    def test_chroma_always_nonnegative(self):
        """Chroma (C) should always be non-negative."""
        rgb = torch.rand(100, 3)
        lch = srgb_to_oklch(rgb)
        assert torch.all(lch[:, 1] >= 0)

    def test_hue_in_range(self):
        """Hue should be in [-pi, pi]."""
        rgb = torch.rand(100, 3)
        lch = srgb_to_oklch(rgb)
        assert torch.all(lch[:, 2] >= -math.pi - 1e-5)
        assert torch.all(lch[:, 2] <= math.pi + 1e-5)


class TestSrgbToOklchGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        # Avoid gray colors where C approaches 0 (gradient issues)
        rgb = (
            torch.rand(5, 3, dtype=torch.float64, requires_grad=True) * 0.8
            + 0.1
        )
        # Add some color variation to avoid near-gray
        rgb = rgb + torch.tensor([[0.2, -0.1, 0.0]], dtype=torch.float64)
        rgb = rgb.clamp(0.05, 0.95)
        rgb.requires_grad_(True)
        assert gradcheck(srgb_to_oklch, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        rgb = (
            torch.rand(4, 4, 3, dtype=torch.float64, requires_grad=True) * 0.8
            + 0.1
        )
        # Add some color variation
        rgb = rgb + torch.tensor([[[0.15, -0.1, 0.05]]], dtype=torch.float64)
        rgb = rgb.clamp(0.05, 0.95)
        rgb.requires_grad_(True)
        assert gradcheck(srgb_to_oklch, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        rgb = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)
        lch = srgb_to_oklch(rgb)
        loss = lch.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.any(torch.isnan(rgb.grad))


class TestSrgbToOklchDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        rgb = torch.rand(10, 3, dtype=torch.float32)
        lch = srgb_to_oklch(rgb)
        assert lch.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        rgb = torch.rand(10, 3, dtype=torch.float64)
        lch = srgb_to_oklch(rgb)
        assert lch.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        rgb = torch.rand(10, 3, dtype=torch.bfloat16)
        lch = srgb_to_oklch(rgb)
        assert lch.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        rgb = torch.rand(10, 3, dtype=torch.float16)
        lch = srgb_to_oklch(rgb)
        assert lch.dtype == torch.float16


class TestSrgbToOklchEdgeCases:
    """Tests for edge cases."""

    def test_hdr_values(self):
        """Test with HDR values > 1."""
        rgb = torch.tensor([1.5, 1.2, 0.8])
        lch = srgb_to_oklch(rgb)
        assert not torch.any(torch.isnan(lch))
        assert lch[0] > 1  # L should exceed 1 for HDR

    def test_negative_values(self):
        """Test with negative values (out of gamut)."""
        rgb = torch.tensor([-0.1, 0.5, 0.5])
        lch = srgb_to_oklch(rgb)
        assert not torch.any(torch.isnan(lch))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        rgb = torch.empty(0, 3)
        lch = srgb_to_oklch(rgb)
        assert lch.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        rgb = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not rgb.is_contiguous()
        lch = srgb_to_oklch(rgb)
        assert lch.shape == rgb.shape


class TestSrgbToOklchConsistency:
    """Tests for consistency with Oklab conversion."""

    def test_consistent_with_oklab(self):
        """Oklch should be consistent with Oklab (L and C=sqrt(a^2+b^2))."""
        from torchscience.color import srgb_to_oklab

        rgb = torch.rand(10, 3)
        lab = srgb_to_oklab(rgb)
        lch = srgb_to_oklch(rgb)

        # L should match
        assert torch.allclose(lch[:, 0], lab[:, 0], atol=1e-5)

        # C should equal sqrt(a^2 + b^2)
        expected_c = torch.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
        assert torch.allclose(lch[:, 1], expected_c, atol=1e-5)

        # h should equal atan2(b, a) for non-zero C
        mask = expected_c > 1e-6
        expected_h = torch.atan2(lab[mask, 2], lab[mask, 1])
        assert torch.allclose(lch[mask, 2], expected_h, atol=1e-5)


class TestSrgbToOklchRoundTrip:
    """Round-trip conversion tests."""

    def test_roundtrip(self):
        """sRGB -> Oklch -> sRGB should recover original."""
        from torchscience.color import oklch_to_srgb

        rgb = torch.rand(100, 3) * 0.9 + 0.05  # Avoid extremes
        lch = srgb_to_oklch(rgb)
        rgb_recovered = oklch_to_srgb(lch)

        assert torch.allclose(rgb, rgb_recovered, atol=1e-5)

    def test_roundtrip_gradients(self):
        """Round-trip should preserve gradients."""
        from torchscience.color import oklch_to_srgb

        rgb = (torch.rand(10, 3) * 0.9 + 0.05).requires_grad_(True)
        lch = srgb_to_oklch(rgb)
        rgb_recovered = oklch_to_srgb(lch)
        loss = rgb_recovered.sum()
        loss.backward()

        assert rgb.grad is not None
        assert not torch.any(torch.isnan(rgb.grad))

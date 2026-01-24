"""Tests for lchuv_to_srgb color conversion."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import lchuv_to_srgb


class TestLchuvToSrgbShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        lch = torch.tensor([50.0, 30.0, 0.5])
        rgb = lchuv_to_srgb(lch)
        assert rgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        lch = torch.randn(10, 3).abs()
        lch[:, 0] = lch[:, 0] * 100  # L in [0, 100]
        lch[:, 1] = lch[:, 1] * 50  # C positive
        rgb = lchuv_to_srgb(lch)
        assert rgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        lch = torch.randn(64, 64, 3).abs()
        lch[..., 0] = lch[..., 0] * 100
        lch[..., 1] = lch[..., 1] * 50
        rgb = lchuv_to_srgb(lch)
        assert rgb.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        lch = torch.randn(10, 32, 32, 3).abs()
        lch[..., 0] = lch[..., 0] * 100
        lch[..., 1] = lch[..., 1] * 50
        rgb = lchuv_to_srgb(lch)
        assert rgb.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            lchuv_to_srgb(torch.randn(10, 4))


class TestLchuvToSrgbKnownValues:
    """Tests for known color conversions."""

    def test_white(self):
        """White: (100, 0, 0) -> (1, 1, 1)."""
        lch = torch.tensor([100.0, 0.0, 0.0])
        rgb = lchuv_to_srgb(lch)
        assert torch.allclose(rgb, torch.tensor([1.0, 1.0, 1.0]), atol=1e-4)

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0, 0)."""
        lch = torch.tensor([0.0, 0.0, 0.0])
        rgb = lchuv_to_srgb(lch)
        assert torch.allclose(rgb, torch.tensor([0.0, 0.0, 0.0]), atol=1e-4)

    def test_hue_invariance_for_gray(self):
        """Hue should not affect gray colors (C=0)."""
        lch1 = torch.tensor([50.0, 0.0, 0.0])
        lch2 = torch.tensor([50.0, 0.0, math.pi])
        lch3 = torch.tensor([50.0, 0.0, -math.pi / 2])

        rgb1 = lchuv_to_srgb(lch1)
        rgb2 = lchuv_to_srgb(lch2)
        rgb3 = lchuv_to_srgb(lch3)

        # All should give the same gray
        assert torch.allclose(rgb1, rgb2, atol=1e-5)
        assert torch.allclose(rgb1, rgb3, atol=1e-5)

    def test_gray_is_achromatic(self):
        """Gray (C=0) should produce R=G=B."""
        lch = torch.tensor([50.0, 0.0, 0.0])
        rgb = lchuv_to_srgb(lch)
        # All channels should be equal for gray
        assert torch.isclose(rgb[0], rgb[1], atol=1e-5)
        assert torch.isclose(rgb[1], rgb[2], atol=1e-5)


class TestLchuvToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        lch = torch.tensor(
            [
                [50.0, 30.0, 0.5],
                [70.0, 40.0, 1.0],
                [30.0, 20.0, -0.5],
                [60.0, 50.0, 2.0],
                [80.0, 25.0, -1.5],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert gradcheck(lchuv_to_srgb, (lch,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        lch = torch.randn(4, 4, 3, dtype=torch.float64)
        lch[..., 0] = lch[..., 0].abs() * 80 + 10  # L in [10, 90]
        lch[..., 1] = lch[..., 1].abs() * 40 + 5  # C in [5, 45]
        lch.requires_grad_(True)
        assert gradcheck(lchuv_to_srgb, (lch,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        lch = torch.tensor([[50.0, 30.0, 0.5]], requires_grad=True)
        rgb = lchuv_to_srgb(lch)
        loss = rgb.sum()
        loss.backward()
        assert lch.grad is not None
        assert not torch.any(torch.isnan(lch.grad))


class TestLchuvToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        lch = torch.tensor([[50.0, 30.0, 0.5]], dtype=torch.float32)
        rgb = lchuv_to_srgb(lch)
        assert rgb.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        lch = torch.tensor([[50.0, 30.0, 0.5]], dtype=torch.float64)
        rgb = lchuv_to_srgb(lch)
        assert rgb.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        lch = torch.tensor([[50.0, 30.0, 0.5]], dtype=torch.bfloat16)
        rgb = lchuv_to_srgb(lch)
        assert rgb.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        lch = torch.tensor([[50.0, 30.0, 0.5]], dtype=torch.float16)
        rgb = lchuv_to_srgb(lch)
        assert rgb.dtype == torch.float16


class TestLchuvToSrgbEdgeCases:
    """Tests for edge cases."""

    def test_high_lightness(self):
        """Test with L > 100 (HDR)."""
        lch = torch.tensor([120.0, 10.0, 0.5])
        rgb = lchuv_to_srgb(lch)
        assert not torch.any(torch.isnan(rgb))
        # At least one channel should exceed 1.0
        assert torch.any(rgb > 1.0)

    def test_high_chroma(self):
        """Test with high chroma (potentially out of gamut)."""
        lch = torch.tensor([50.0, 150.0, 0.5])
        rgb = lchuv_to_srgb(lch)
        assert not torch.any(torch.isnan(rgb))

    def test_negative_hue(self):
        """Test with negative hue."""
        lch = torch.tensor([50.0, 30.0, -math.pi / 2])
        rgb = lchuv_to_srgb(lch)
        assert not torch.any(torch.isnan(rgb))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        lch = torch.empty(0, 3)
        rgb = lchuv_to_srgb(lch)
        assert rgb.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        lch = torch.randn(10, 10, 3)[:, ::2, :]  # Non-contiguous
        lch[..., 0] = lch[..., 0].abs() * 100
        lch[..., 1] = lch[..., 1].abs() * 50
        assert not lch.is_contiguous()
        rgb = lchuv_to_srgb(lch)
        assert rgb.shape == lch.shape


class TestLchuvToSrgbConsistency:
    """Tests for consistency with LUV conversion."""

    def test_consistent_with_luv(self):
        """LCH -> sRGB should match LUV -> sRGB for corresponding values."""
        from torchscience.color import luv_to_srgb

        # Create LCH values
        L = torch.rand(10) * 80 + 10
        C = torch.rand(10) * 50 + 5
        h = torch.rand(10) * 2 * math.pi - math.pi

        lch = torch.stack([L, C, h], dim=-1)

        # Convert LCH to LUV manually
        u = C * torch.cos(h)
        v = C * torch.sin(h)
        luv = torch.stack([L, u, v], dim=-1)

        # Both should give same RGB
        rgb_from_lch = lchuv_to_srgb(lch)
        rgb_from_luv = luv_to_srgb(luv)

        assert torch.allclose(rgb_from_lch, rgb_from_luv, atol=1e-5)


class TestLchuvToSrgbRoundTrip:
    """Round-trip conversion tests."""

    def test_roundtrip(self):
        """LCHuv -> sRGB -> LCHuv should recover original (for in-gamut colors)."""
        from torchscience.color import srgb_to_lchuv

        # Start with valid sRGB colors to ensure in-gamut
        rgb = torch.rand(100, 3) * 0.9 + 0.05
        lch = srgb_to_lchuv(rgb)
        rgb_converted = lchuv_to_srgb(lch)
        lch_recovered = srgb_to_lchuv(rgb_converted)

        # L and C should match closely
        assert torch.allclose(lch[:, 0], lch_recovered[:, 0], atol=1e-4)
        assert torch.allclose(lch[:, 1], lch_recovered[:, 1], atol=1e-4)

        # For hue, handle the C=0 case where hue is undefined
        mask = lch[:, 1] > 1e-3
        assert torch.allclose(lch[mask, 2], lch_recovered[mask, 2], atol=1e-4)

    def test_roundtrip_gradients(self):
        """Round-trip should preserve gradients."""
        from torchscience.color import srgb_to_lchuv

        lch = torch.tensor(
            [
                [50.0, 30.0, 0.5],
                [70.0, 40.0, 1.0],
            ],
            requires_grad=True,
        )
        rgb = lchuv_to_srgb(lch)
        lch_recovered = srgb_to_lchuv(rgb)
        loss = lch_recovered.sum()
        loss.backward()

        assert lch.grad is not None
        assert not torch.any(torch.isnan(lch.grad))

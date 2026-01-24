"""Tests for srgb_to_lchuv color conversion."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_lchuv


class TestSrgbToLchuvShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        lch = srgb_to_lchuv(rgb)
        assert lch.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        rgb = torch.randn(10, 3).abs()
        lch = srgb_to_lchuv(rgb)
        assert lch.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        rgb = torch.randn(64, 64, 3).abs()
        lch = srgb_to_lchuv(rgb)
        assert lch.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        rgb = torch.randn(10, 32, 32, 3).abs()
        lch = srgb_to_lchuv(rgb)
        assert lch.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            srgb_to_lchuv(torch.randn(10, 4))


class TestSrgbToLchuvKnownValues:
    """Tests for known color conversions."""

    def test_white_d65(self):
        """D65 white point: (1, 1, 1) -> (100, 0, 0)."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        lch = srgb_to_lchuv(rgb)
        # White in LCH should be L=100, C=0, h=0 (hue undefined when C=0)
        assert torch.isclose(lch[0], torch.tensor(100.0), atol=1e-3)
        assert torch.isclose(lch[1], torch.tensor(0.0), atol=1e-3)
        # h is undefined when C=0, but we set it to 0

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0, 0)."""
        rgb = torch.tensor([0.0, 0.0, 0.0])
        lch = srgb_to_lchuv(rgb)
        # Black should have L=0, C=0
        assert torch.isclose(lch[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(lch[1], torch.tensor(0.0), atol=1e-4)

    def test_red(self):
        """Pure red sRGB primary."""
        rgb = torch.tensor([1.0, 0.0, 0.0])
        lch = srgb_to_lchuv(rgb)
        # Reference: L* ~ 53.23 (same as Lab), C and h are different in LUV space
        assert torch.isclose(lch[0], torch.tensor(53.2329), atol=0.1)
        # C should be positive
        assert lch[1] > 0
        # Hue for red in LUV space
        assert not torch.isnan(lch[2])

    def test_green(self):
        """Pure green sRGB primary."""
        rgb = torch.tensor([0.0, 1.0, 0.0])
        lch = srgb_to_lchuv(rgb)
        # Reference: L* ~ 87.74 (same as Lab)
        assert torch.isclose(lch[0], torch.tensor(87.7370), atol=0.1)
        # C should be positive
        assert lch[1] > 0

    def test_blue(self):
        """Pure blue sRGB primary."""
        rgb = torch.tensor([0.0, 0.0, 1.0])
        lch = srgb_to_lchuv(rgb)
        # Reference: L* ~ 32.30 (same as Lab)
        assert torch.isclose(lch[0], torch.tensor(32.3026), atol=0.1)

    def test_mid_gray(self):
        """Mid gray (0.5, 0.5, 0.5) - should have C=0."""
        rgb = torch.tensor([0.5, 0.5, 0.5])
        lch = srgb_to_lchuv(rgb)
        # Gray should have C=0
        assert torch.isclose(lch[1], torch.tensor(0.0), atol=1e-3)
        # L should be between 0 and 100
        assert lch[0] > 0 and lch[0] < 100

    def test_chroma_always_nonnegative(self):
        """Chroma (C) should always be non-negative."""
        rgb = torch.rand(100, 3)
        lch = srgb_to_lchuv(rgb)
        assert torch.all(lch[:, 1] >= 0)

    def test_hue_in_range(self):
        """Hue should be in [-pi, pi]."""
        rgb = torch.rand(100, 3)
        lch = srgb_to_lchuv(rgb)
        assert torch.all(lch[:, 2] >= -math.pi - 1e-5)
        assert torch.all(lch[:, 2] <= math.pi + 1e-5)


class TestSrgbToLchuvGradients:
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
        assert gradcheck(srgb_to_lchuv, (rgb,), eps=1e-6, atol=1e-4)

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
        assert gradcheck(srgb_to_lchuv, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        rgb = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)
        lch = srgb_to_lchuv(rgb)
        loss = lch.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.any(torch.isnan(rgb.grad))


class TestSrgbToLchuvDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        rgb = torch.rand(10, 3, dtype=torch.float32)
        lch = srgb_to_lchuv(rgb)
        assert lch.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        rgb = torch.rand(10, 3, dtype=torch.float64)
        lch = srgb_to_lchuv(rgb)
        assert lch.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        rgb = torch.rand(10, 3, dtype=torch.bfloat16)
        lch = srgb_to_lchuv(rgb)
        assert lch.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        rgb = torch.rand(10, 3, dtype=torch.float16)
        lch = srgb_to_lchuv(rgb)
        assert lch.dtype == torch.float16


class TestSrgbToLchuvEdgeCases:
    """Tests for edge cases."""

    def test_hdr_values(self):
        """Test with HDR values > 1."""
        rgb = torch.tensor([1.5, 1.2, 0.8])
        lch = srgb_to_lchuv(rgb)
        assert not torch.any(torch.isnan(lch))
        assert lch[0] > 100  # L should exceed 100 for HDR

    def test_negative_values(self):
        """Test with negative values (out of gamut)."""
        rgb = torch.tensor([-0.1, 0.5, 0.5])
        lch = srgb_to_lchuv(rgb)
        assert not torch.any(torch.isnan(lch))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        rgb = torch.empty(0, 3)
        lch = srgb_to_lchuv(rgb)
        assert lch.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        rgb = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not rgb.is_contiguous()
        lch = srgb_to_lchuv(rgb)
        assert lch.shape == rgb.shape


class TestSrgbToLchuvConsistency:
    """Tests for consistency with LUV conversion."""

    def test_consistent_with_luv(self):
        """LCH should be consistent with LUV (L and C=sqrt(u^2+v^2))."""
        from torchscience.color import srgb_to_luv

        rgb = torch.rand(10, 3)
        luv = srgb_to_luv(rgb)
        lch = srgb_to_lchuv(rgb)

        # L should match
        assert torch.allclose(lch[:, 0], luv[:, 0], atol=1e-5)

        # C should equal sqrt(u^2 + v^2)
        expected_c = torch.sqrt(luv[:, 1] ** 2 + luv[:, 2] ** 2)
        assert torch.allclose(lch[:, 1], expected_c, atol=1e-5)

        # h should equal atan2(v, u) for non-zero C
        mask = expected_c > 1e-6
        expected_h = torch.atan2(luv[mask, 2], luv[mask, 1])
        assert torch.allclose(lch[mask, 2], expected_h, atol=1e-5)


class TestSrgbToLchuvRoundTrip:
    """Round-trip conversion tests."""

    def test_roundtrip(self):
        """sRGB -> LCHuv -> sRGB should recover original."""
        from torchscience.color import lchuv_to_srgb

        rgb = torch.rand(100, 3) * 0.9 + 0.05  # Avoid extremes
        lch = srgb_to_lchuv(rgb)
        rgb_recovered = lchuv_to_srgb(lch)

        assert torch.allclose(rgb, rgb_recovered, atol=1e-5)

    def test_roundtrip_gradients(self):
        """Round-trip should preserve gradients."""
        from torchscience.color import lchuv_to_srgb

        rgb = (torch.rand(10, 3) * 0.9 + 0.05).requires_grad_(True)
        lch = srgb_to_lchuv(rgb)
        rgb_recovered = lchuv_to_srgb(lch)
        loss = rgb_recovered.sum()
        loss.backward()

        assert rgb.grad is not None
        assert not torch.any(torch.isnan(rgb.grad))

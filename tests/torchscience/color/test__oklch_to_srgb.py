"""Tests for oklch_to_srgb color conversion."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import oklch_to_srgb


class TestOklchToSrgbShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        lch = torch.tensor([0.5, 0.1, 0.5])
        rgb = oklch_to_srgb(lch)
        assert rgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        lch = torch.zeros(10, 3)
        lch[:, 0] = torch.rand(10)  # L in [0, 1]
        lch[:, 1] = torch.rand(10) * 0.2  # C in [0, 0.2]
        lch[:, 2] = torch.rand(10) * 2 * math.pi - math.pi  # h in [-pi, pi]
        rgb = oklch_to_srgb(lch)
        assert rgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        lch = torch.zeros(64, 64, 3)
        lch[:, :, 0] = 0.5
        lch[:, :, 1] = 0.1
        lch[:, :, 2] = 0.0
        rgb = oklch_to_srgb(lch)
        assert rgb.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        lch = torch.zeros(10, 32, 32, 3)
        lch[:, :, :, 0] = 0.5
        lch[:, :, :, 1] = 0.1
        rgb = oklch_to_srgb(lch)
        assert rgb.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            oklch_to_srgb(torch.randn(10, 4))


class TestOklchToSrgbKnownValues:
    """Tests for known color conversions."""

    def test_white(self):
        """White: L=1, C=0 -> (1, 1, 1)."""
        lch = torch.tensor([1.0, 0.0, 0.0])
        rgb = oklch_to_srgb(lch)
        assert torch.allclose(rgb, torch.tensor([1.0, 1.0, 1.0]), atol=1e-3)

    def test_black(self):
        """Black: L=0, C=0 -> (0, 0, 0)."""
        lch = torch.tensor([0.0, 0.0, 0.0])
        rgb = oklch_to_srgb(lch)
        assert torch.allclose(rgb, torch.tensor([0.0, 0.0, 0.0]), atol=1e-3)

    def test_red_approximate(self):
        """Approximate red: known Oklab values converted to Oklch."""
        # Red in Oklab: L~0.628, a~0.2249, b~0.1262
        # Oklch: L~0.628, C~sqrt(0.2249^2+0.1262^2)~0.258, h~atan2(0.1262,0.2249)~0.51
        L = 0.6280
        a = 0.2249
        b = 0.1262
        C = math.sqrt(a**2 + b**2)
        h = math.atan2(b, a)
        lch = torch.tensor([L, C, h])
        rgb = oklch_to_srgb(lch)
        # Should be approximately red
        assert rgb[0] > 0.9  # R should be high
        assert rgb[1] < 0.1  # G should be low
        assert rgb[2] < 0.1  # B should be low

    def test_mid_gray(self):
        """Mid gray: L=0.5, C=0 -> gray RGB."""
        lch = torch.tensor([0.5, 0.0, 0.0])
        rgb = oklch_to_srgb(lch)
        # Should be roughly gray
        assert torch.allclose(rgb[0], rgb[1], atol=1e-3)
        assert torch.allclose(rgb[1], rgb[2], atol=1e-3)

    def test_hue_invariance_at_zero_chroma(self):
        """At C=0, any hue should give the same result (gray)."""
        L = 0.5
        C = 0.0
        rgb_h0 = oklch_to_srgb(torch.tensor([L, C, 0.0]))
        rgb_h1 = oklch_to_srgb(torch.tensor([L, C, math.pi / 2]))
        rgb_h2 = oklch_to_srgb(torch.tensor([L, C, math.pi]))
        rgb_h3 = oklch_to_srgb(torch.tensor([L, C, -math.pi / 2]))

        assert torch.allclose(rgb_h0, rgb_h1, atol=1e-5)
        assert torch.allclose(rgb_h1, rgb_h2, atol=1e-5)
        assert torch.allclose(rgb_h2, rgb_h3, atol=1e-5)


class TestOklchToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        lch = torch.zeros(5, 3, dtype=torch.float64, requires_grad=True)
        lch_data = lch.data
        lch_data[:, 0] = (
            torch.rand(5, dtype=torch.float64) * 0.8 + 0.1
        )  # L in [0.1, 0.9]
        lch_data[:, 1] = (
            torch.rand(5, dtype=torch.float64) * 0.15 + 0.05
        )  # C in [0.05, 0.2]
        lch_data[:, 2] = (
            torch.rand(5, dtype=torch.float64) * 2 * math.pi - math.pi
        )
        lch = lch_data.clone().requires_grad_(True)
        assert gradcheck(oklch_to_srgb, (lch,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        lch = torch.zeros(4, 4, 3, dtype=torch.float64)
        lch[:, :, 0] = torch.rand(4, 4, dtype=torch.float64) * 0.8 + 0.1
        lch[:, :, 1] = torch.rand(4, 4, dtype=torch.float64) * 0.15 + 0.05
        lch[:, :, 2] = (
            torch.rand(4, 4, dtype=torch.float64) * 2 * math.pi - math.pi
        )
        lch = lch.requires_grad_(True)
        assert gradcheck(oklch_to_srgb, (lch,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        lch = torch.tensor([[0.5, 0.1, 0.5]], requires_grad=True)
        rgb = oklch_to_srgb(lch)
        loss = rgb.sum()
        loss.backward()
        assert lch.grad is not None
        assert not torch.any(torch.isnan(lch.grad))

    def test_gradient_at_zero_chroma(self):
        """Test gradient behavior at C=0."""
        # At C=0, gradients w.r.t. h should be zero (hue doesn't affect output)
        lch = torch.tensor(
            [[0.5, 0.0, 0.5]], dtype=torch.float64, requires_grad=True
        )
        rgb = oklch_to_srgb(lch)
        loss = rgb.sum()
        loss.backward()
        # Gradient w.r.t. h (index 2) should be 0 when C=0
        assert torch.isclose(
            lch.grad[0, 2], torch.tensor(0.0, dtype=torch.float64), atol=1e-6
        )


class TestOklchToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        lch = torch.tensor([[0.5, 0.1, 0.5]], dtype=torch.float32)
        rgb = oklch_to_srgb(lch)
        assert rgb.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        lch = torch.tensor([[0.5, 0.1, 0.5]], dtype=torch.float64)
        rgb = oklch_to_srgb(lch)
        assert rgb.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        lch = torch.tensor([[0.5, 0.1, 0.5]], dtype=torch.bfloat16)
        rgb = oklch_to_srgb(lch)
        assert rgb.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        lch = torch.tensor([[0.5, 0.1, 0.5]], dtype=torch.float16)
        rgb = oklch_to_srgb(lch)
        assert rgb.dtype == torch.float16


class TestOklchToSrgbEdgeCases:
    """Tests for edge cases."""

    def test_high_lightness(self):
        """Test with L > 1 (HDR-like)."""
        lch = torch.tensor([1.2, 0.05, 0.0])
        rgb = oklch_to_srgb(lch)
        assert not torch.any(torch.isnan(rgb))
        assert rgb[0] > 1.0  # Should be > 1 for HDR

    def test_negative_lightness(self):
        """Test with negative L (out of spec)."""
        lch = torch.tensor([-0.1, 0.05, 0.0])
        rgb = oklch_to_srgb(lch)
        assert not torch.any(torch.isnan(rgb))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        lch = torch.empty(0, 3)
        rgb = oklch_to_srgb(lch)
        assert rgb.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        lch = torch.zeros(10, 10, 3)
        lch[:, :, 0] = 0.5
        lch[:, :, 1] = 0.1
        lch = lch[:, ::2, :]  # Non-contiguous
        assert not lch.is_contiguous()
        rgb = oklch_to_srgb(lch)
        assert rgb.shape == lch.shape

    def test_large_chroma(self):
        """Test with very large chroma (out of gamut)."""
        lch = torch.tensor([0.5, 0.5, 0.0])  # C=0.5 is very saturated
        rgb = oklch_to_srgb(lch)
        assert not torch.any(torch.isnan(rgb))
        # Result will be out of gamut (some values < 0 or > 1)


class TestOklchToSrgbConsistency:
    """Tests for consistency with Oklab conversion."""

    def test_consistent_with_oklab(self):
        """Oklch -> sRGB should match Oklab -> sRGB when converted properly."""
        from torchscience.color import oklab_to_srgb

        # Create some Oklch values
        L = torch.rand(10) * 0.8 + 0.1  # L in [0.1, 0.9]
        C = torch.rand(10) * 0.15 + 0.05  # C in [0.05, 0.2]
        h = torch.rand(10) * 2 * math.pi - math.pi

        lch = torch.stack([L, C, h], dim=-1)

        # Convert to Oklab manually
        a = C * torch.cos(h)
        b = C * torch.sin(h)
        lab = torch.stack([L, a, b], dim=-1)

        # Compare outputs
        rgb_from_lch = oklch_to_srgb(lch)
        rgb_from_lab = oklab_to_srgb(lab)

        assert torch.allclose(rgb_from_lch, rgb_from_lab, atol=1e-5)


class TestOklchToSrgbRoundTrip:
    """Round-trip conversion tests."""

    def test_roundtrip(self):
        """Oklch -> sRGB -> Oklch should recover original."""
        from torchscience.color import srgb_to_oklch

        # Start with valid Oklch values that produce in-gamut RGB
        L = torch.rand(100) * 0.8 + 0.1  # L in [0.1, 0.9]
        C = torch.rand(100) * 0.1 + 0.02  # C in [0.02, 0.12] (conservative)
        h = torch.rand(100) * 2 * math.pi - math.pi

        lch = torch.stack([L, C, h], dim=-1)

        rgb = oklch_to_srgb(lch)
        # Clamp to valid range for round-trip
        rgb = rgb.clamp(0.01, 0.99)
        lch_recovered = srgb_to_oklch(rgb)
        rgb_recovered = oklch_to_srgb(lch_recovered)

        # RGB should match after round-trip through clamped values
        assert torch.allclose(rgb, rgb_recovered, atol=1e-5)

    def test_roundtrip_gradients(self):
        """Round-trip should preserve gradients."""
        from torchscience.color import srgb_to_oklch

        L = torch.rand(10) * 0.6 + 0.2
        C = torch.rand(10) * 0.1 + 0.02
        h = torch.rand(10) * 2 * math.pi - math.pi
        lch = torch.stack([L, C, h], dim=-1).requires_grad_(True)

        rgb = oklch_to_srgb(lch)
        lch_recovered = srgb_to_oklch(rgb)
        loss = lch_recovered.sum()
        loss.backward()

        assert lch.grad is not None
        assert not torch.any(torch.isnan(lch.grad))

    def test_srgb_roundtrip(self):
        """sRGB -> Oklch -> sRGB should recover original."""
        from torchscience.color import srgb_to_oklch

        rgb = torch.rand(100, 3) * 0.9 + 0.05  # Avoid extremes
        lch = srgb_to_oklch(rgb)
        rgb_recovered = oklch_to_srgb(lch)

        assert torch.allclose(rgb, rgb_recovered, atol=1e-5)

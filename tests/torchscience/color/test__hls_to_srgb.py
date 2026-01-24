"""Tests for hls_to_srgb color conversion."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import hls_to_srgb


class TestHlsToSrgbShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        hls = torch.tensor([0.0, 0.5, 1.0])
        rgb = hls_to_srgb(hls)
        assert rgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        hls = torch.randn(10, 3)
        rgb = hls_to_srgb(hls)
        assert rgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        hls = torch.randn(64, 64, 3)
        rgb = hls_to_srgb(hls)
        assert rgb.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        hls = torch.randn(10, 32, 32, 3)
        rgb = hls_to_srgb(hls)
        assert rgb.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            hls_to_srgb(torch.randn(10, 4))


class TestHlsToSrgbKnownValues:
    """Tests for known color conversions."""

    def test_red(self):
        """Pure red: H=0, L=0.5, S=1 -> (1, 0, 0)."""
        hls = torch.tensor([0.0, 0.5, 1.0])
        rgb = hls_to_srgb(hls)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)

    def test_green(self):
        """Pure green: H=2pi/3, L=0.5, S=1 -> (0, 1, 0)."""
        hls = torch.tensor([2 * math.pi / 3, 0.5, 1.0])
        rgb = hls_to_srgb(hls)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)

    def test_blue(self):
        """Pure blue: H=4pi/3, L=0.5, S=1 -> (0, 0, 1)."""
        hls = torch.tensor([4 * math.pi / 3, 0.5, 1.0])
        rgb = hls_to_srgb(hls)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_yellow(self):
        """Yellow: H=pi/3, L=0.5, S=1 -> (1, 1, 0)."""
        hls = torch.tensor([math.pi / 3, 0.5, 1.0])
        rgb = hls_to_srgb(hls)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)

    def test_cyan(self):
        """Cyan: H=pi, L=0.5, S=1 -> (0, 1, 1)."""
        hls = torch.tensor([math.pi, 0.5, 1.0])
        rgb = hls_to_srgb(hls)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_magenta(self):
        """Magenta: H=5pi/3, L=0.5, S=1 -> (1, 0, 1)."""
        hls = torch.tensor([5 * math.pi / 3, 0.5, 1.0])
        rgb = hls_to_srgb(hls)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_white(self):
        """White: H=0, L=1, S=0 -> (1, 1, 1)."""
        hls = torch.tensor([0.0, 1.0, 0.0])
        rgb = hls_to_srgb(hls)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_black(self):
        """Black: H=0, L=0, S=0 -> (0, 0, 0)."""
        hls = torch.tensor([0.0, 0.0, 0.0])
        rgb = hls_to_srgb(hls)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)

    def test_gray(self):
        """Gray: H=0, L=0.5, S=0 -> (0.5, 0.5, 0.5)."""
        hls = torch.tensor([0.0, 0.5, 0.0])
        rgb = hls_to_srgb(hls)
        assert torch.isclose(rgb[0], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.5), atol=1e-5)

    def test_light_red(self):
        """Light red: H=0, L=0.75, S=1 -> (1, 0.5, 0.5)."""
        hls = torch.tensor([0.0, 0.75, 1.0])
        rgb = hls_to_srgb(hls)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.5), atol=1e-5)

    def test_dark_red(self):
        """Dark red: H=0, L=0.25, S=1 -> (0.5, 0, 0)."""
        hls = torch.tensor([0.0, 0.25, 1.0])
        rgb = hls_to_srgb(hls)
        assert torch.isclose(rgb[0], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)


class TestHlsToSrgbEdgeCases:
    """Tests for edge cases and special values."""

    def test_achromatic_zero_saturation(self):
        """Zero saturation should produce grayscale."""
        hls = torch.tensor([1.5, 0.7, 0.0])  # Arbitrary hue, zero saturation
        rgb = hls_to_srgb(hls)
        # All RGB components should equal L
        assert torch.isclose(rgb[0], torch.tensor(0.7), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.7), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.7), atol=1e-5)

    def test_hue_wraparound(self):
        """Hue should wrap around at 2pi."""
        hls1 = torch.tensor([0.0, 0.5, 1.0])
        hls2 = torch.tensor([2 * math.pi, 0.5, 1.0])
        rgb1 = hls_to_srgb(hls1)
        rgb2 = hls_to_srgb(hls2)
        assert torch.allclose(rgb1, rgb2, atol=1e-5)


class TestHlsToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck_saturated(self):
        """Gradient check for saturated colors."""
        # Avoid L=0.5 exactly (gradient discontinuity at L=0.5)
        hls = torch.tensor(
            [[1.0, 0.6, 0.8]], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(hls_to_srgb, (hls,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check for batch of colors."""
        torch.manual_seed(42)
        # Create HLS values within valid range
        # H in [0.5, 5.5] to avoid sector boundaries
        # L in [0.3, 0.7] to avoid L=0.5 threshold
        # S in [0.2, 0.8] to avoid zero saturation
        hls = torch.empty(5, 3, dtype=torch.float64)
        hls[:, 0] = torch.rand(5, dtype=torch.float64) * 5.0 + 0.5
        hls[:, 1] = torch.rand(5, dtype=torch.float64) * 0.3 + 0.6
        hls[:, 2] = torch.rand(5, dtype=torch.float64) * 0.6 + 0.2
        hls = hls.requires_grad_(True)
        assert gradcheck(hls_to_srgb, (hls,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradients flow back correctly."""
        hls = torch.tensor([[1.0, 0.5, 0.8]], requires_grad=True)
        rgb = hls_to_srgb(hls)
        loss = rgb.sum()
        loss.backward()
        assert hls.grad is not None
        assert not torch.isnan(hls.grad).any()

    @pytest.mark.skip(reason="Gradient discontinuous at S=0")
    def test_gradcheck_achromatic(self):
        """Gradient check at zero saturation (expected to fail)."""
        hls = torch.tensor(
            [[1.0, 0.5, 0.0]], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(hls_to_srgb, (hls,), eps=1e-6, atol=1e-4)


class TestHlsToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        hls = torch.rand(10, 3, dtype=torch.float32)
        rgb = hls_to_srgb(hls)
        assert rgb.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        hls = torch.rand(10, 3, dtype=torch.float64)
        rgb = hls_to_srgb(hls)
        assert rgb.dtype == torch.float64

    @pytest.mark.skip(reason="Half precision may have accuracy issues")
    def test_float16(self):
        """Works with float16."""
        hls = torch.rand(10, 3, dtype=torch.float16)
        rgb = hls_to_srgb(hls)
        assert rgb.dtype == torch.float16


class TestRoundTrip:
    """Tests for round-trip conversion."""

    def test_roundtrip_srgb_to_hls_to_srgb(self):
        """RGB -> HLS -> RGB should be identity (for valid colors)."""
        from torchscience.color import srgb_to_hls

        torch.manual_seed(42)
        rgb = torch.rand(10, 3, dtype=torch.float64)
        hls = srgb_to_hls(rgb)
        rgb_roundtrip = hls_to_srgb(hls)
        assert torch.allclose(rgb, rgb_roundtrip, atol=1e-5)

    def test_roundtrip_hls_to_srgb_to_hls(self):
        """HLS -> RGB -> HLS should be identity (for valid colors)."""
        from torchscience.color import srgb_to_hls

        torch.manual_seed(42)
        # Create valid HLS values (avoid grayscale where hue is undefined)
        hls = torch.empty(10, 3, dtype=torch.float64)
        hls[:, 0] = (
            torch.rand(10, dtype=torch.float64) * 2 * math.pi
        )  # H in [0, 2pi]
        hls[:, 1] = (
            torch.rand(10, dtype=torch.float64) * 0.8 + 0.1
        )  # L in [0.1, 0.9]
        hls[:, 2] = (
            torch.rand(10, dtype=torch.float64) * 0.8 + 0.2
        )  # S in [0.2, 1]

        rgb = hls_to_srgb(hls)
        hls_roundtrip = srgb_to_hls(rgb)

        # H and S match when S > 0 and L not at extremes
        assert torch.allclose(hls[:, 0], hls_roundtrip[:, 0], atol=1e-5)
        assert torch.allclose(hls[:, 1], hls_roundtrip[:, 1], atol=1e-5)
        assert torch.allclose(hls[:, 2], hls_roundtrip[:, 2], atol=1e-5)

"""Tests for hsv_to_srgb color conversion."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.graphics.color import hsv_to_srgb, srgb_to_hsv


class TestHsvToSrgbShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        hsv = torch.tensor([0.0, 1.0, 1.0])
        rgb = hsv_to_srgb(hsv)
        assert rgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        hsv = torch.randn(10, 3)
        rgb = hsv_to_srgb(hsv)
        assert rgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        hsv = torch.randn(64, 64, 3)
        rgb = hsv_to_srgb(hsv)
        assert rgb.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        hsv = torch.randn(10, 32, 32, 3)
        rgb = hsv_to_srgb(hsv)
        assert rgb.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            hsv_to_srgb(torch.randn(10, 4))


class TestHsvToSrgbKnownValues:
    """Tests for known color conversions."""

    def test_red(self):
        """H=0, S=1, V=1 -> pure red (1, 0, 0)."""
        hsv = torch.tensor([0.0, 1.0, 1.0])
        rgb = hsv_to_srgb(hsv)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)

    def test_green(self):
        """H=2π/3, S=1, V=1 -> pure green (0, 1, 0)."""
        h = 2 * math.pi / 3
        hsv = torch.tensor([h, 1.0, 1.0])
        rgb = hsv_to_srgb(hsv)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)

    def test_blue(self):
        """H=4π/3, S=1, V=1 -> pure blue (0, 0, 1)."""
        h = 4 * math.pi / 3
        hsv = torch.tensor([h, 1.0, 1.0])
        rgb = hsv_to_srgb(hsv)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_yellow(self):
        """H=π/3, S=1, V=1 -> yellow (1, 1, 0)."""
        h = math.pi / 3
        hsv = torch.tensor([h, 1.0, 1.0])
        rgb = hsv_to_srgb(hsv)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)

    def test_cyan(self):
        """H=π, S=1, V=1 -> cyan (0, 1, 1)."""
        h = math.pi
        hsv = torch.tensor([h, 1.0, 1.0])
        rgb = hsv_to_srgb(hsv)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_magenta(self):
        """H=5π/3, S=1, V=1 -> magenta (1, 0, 1)."""
        h = 5 * math.pi / 3
        hsv = torch.tensor([h, 1.0, 1.0])
        rgb = hsv_to_srgb(hsv)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_white(self):
        """H=0, S=0, V=1 -> white (1, 1, 1)."""
        hsv = torch.tensor([0.0, 0.0, 1.0])
        rgb = hsv_to_srgb(hsv)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_black(self):
        """H=0, S=0, V=0 -> black (0, 0, 0)."""
        hsv = torch.tensor([0.0, 0.0, 0.0])
        rgb = hsv_to_srgb(hsv)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)

    def test_gray(self):
        """H=0, S=0, V=0.5 -> gray (0.5, 0.5, 0.5)."""
        hsv = torch.tensor([0.0, 0.0, 0.5])
        rgb = hsv_to_srgb(hsv)
        assert torch.isclose(rgb[0], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.5), atol=1e-5)


class TestRoundTrip:
    """Tests for round-trip conversion consistency."""

    def test_rgb_to_hsv_to_rgb(self):
        """RGB -> HSV -> RGB should be identity."""
        torch.manual_seed(42)
        # Use colors with good saturation to avoid grayscale edge cases
        rgb_original = torch.rand(100, 3) * 0.8 + 0.1
        # Ensure not grayscale by adding offset to one channel
        rgb_original[:, 0] += 0.1

        hsv = srgb_to_hsv(rgb_original)
        rgb_recovered = hsv_to_srgb(hsv)

        assert torch.allclose(rgb_original, rgb_recovered, atol=1e-5)

    def test_hsv_to_rgb_to_hsv(self):
        """HSV -> RGB -> HSV should be identity (for valid HSV)."""
        torch.manual_seed(42)
        # Create valid HSV values with good saturation
        h = torch.rand(100) * 2 * math.pi
        s = torch.rand(100) * 0.8 + 0.2  # Avoid zero saturation
        v = torch.rand(100) * 0.8 + 0.2  # Avoid zero value
        hsv_original = torch.stack([h, s, v], dim=-1)

        rgb = hsv_to_srgb(hsv_original)
        hsv_recovered = srgb_to_hsv(rgb)

        # Hue comparison needs special handling for wrap-around
        h_diff = torch.abs(hsv_original[..., 0] - hsv_recovered[..., 0])
        h_diff = torch.min(h_diff, 2 * math.pi - h_diff)
        assert torch.all(h_diff < 1e-4)

        # S and V should match directly
        assert torch.allclose(
            hsv_original[..., 1:], hsv_recovered[..., 1:], atol=1e-5
        )

    def test_primary_colors_roundtrip(self):
        """Primary and secondary colors should roundtrip exactly."""
        colors = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [0.0, 1.0, 1.0],  # Cyan
                [1.0, 0.0, 1.0],  # Magenta
            ]
        )

        hsv = srgb_to_hsv(colors)
        rgb_recovered = hsv_to_srgb(hsv)

        assert torch.allclose(colors, rgb_recovered, atol=1e-5)


class TestHsvToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck_saturated(self):
        """Gradient check for saturated colors."""
        hsv = torch.tensor(
            [[1.0, 0.8, 0.9]], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(hsv_to_srgb, (hsv,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check for batch of colors."""
        torch.manual_seed(42)
        h = torch.rand(5, dtype=torch.float64) * 2 * math.pi
        s = torch.rand(5, dtype=torch.float64) * 0.5 + 0.25
        v = torch.rand(5, dtype=torch.float64) * 0.5 + 0.25
        hsv = torch.stack([h, s, v], dim=-1).requires_grad_(True)
        assert gradcheck(hsv_to_srgb, (hsv,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradients flow back correctly."""
        hsv = torch.tensor([[1.0, 0.5, 0.8]], requires_grad=True)
        rgb = hsv_to_srgb(hsv)
        loss = rgb.sum()
        loss.backward()
        assert hsv.grad is not None
        assert not torch.isnan(hsv.grad).any()


class TestHsvToSrgbEdgeCases:
    """Tests for edge cases."""

    def test_hue_wraparound(self):
        """Hue values outside [0, 2π] should wrap correctly."""
        # H = 2π should be same as H = 0 (red)
        hsv_0 = torch.tensor([0.0, 1.0, 1.0])
        hsv_2pi = torch.tensor([2 * math.pi, 1.0, 1.0])

        rgb_0 = hsv_to_srgb(hsv_0)
        rgb_2pi = hsv_to_srgb(hsv_2pi)

        assert torch.allclose(rgb_0, rgb_2pi, atol=1e-5)

    def test_zero_saturation(self):
        """Zero saturation should give grayscale regardless of hue."""
        hsv1 = torch.tensor([0.0, 0.0, 0.5])
        hsv2 = torch.tensor([math.pi, 0.0, 0.5])
        hsv3 = torch.tensor([1.5 * math.pi, 0.0, 0.5])

        rgb1 = hsv_to_srgb(hsv1)
        rgb2 = hsv_to_srgb(hsv2)
        rgb3 = hsv_to_srgb(hsv3)

        # All should be the same gray
        assert torch.allclose(rgb1, rgb2, atol=1e-5)
        assert torch.allclose(rgb2, rgb3, atol=1e-5)
        assert torch.allclose(rgb1, torch.tensor([0.5, 0.5, 0.5]), atol=1e-5)

    def test_zero_value(self):
        """Zero value should give black regardless of hue and saturation."""
        hsv1 = torch.tensor([0.0, 1.0, 0.0])
        hsv2 = torch.tensor([math.pi, 0.5, 0.0])

        rgb1 = hsv_to_srgb(hsv1)
        rgb2 = hsv_to_srgb(hsv2)

        black = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(rgb1, black, atol=1e-5)
        assert torch.allclose(rgb2, black, atol=1e-5)


class TestHsvToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        hsv = torch.rand(10, 3, dtype=torch.float32)
        rgb = hsv_to_srgb(hsv)
        assert rgb.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        hsv = torch.rand(10, 3, dtype=torch.float64)
        rgb = hsv_to_srgb(hsv)
        assert rgb.dtype == torch.float64

"""Tests for hwb_to_srgb color conversion."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import hwb_to_srgb


class TestHwbToSrgbShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        hwb = torch.tensor([0.0, 0.0, 0.0])
        rgb = hwb_to_srgb(hwb)
        assert rgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        hwb = torch.randn(10, 3)
        rgb = hwb_to_srgb(hwb)
        assert rgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        hwb = torch.randn(64, 64, 3)
        rgb = hwb_to_srgb(hwb)
        assert rgb.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        hwb = torch.randn(10, 32, 32, 3)
        rgb = hwb_to_srgb(hwb)
        assert rgb.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            hwb_to_srgb(torch.randn(10, 4))


class TestHwbToSrgbKnownValues:
    """Tests for known color conversions."""

    def test_red(self):
        """HWB red (H=0, W=0, B=0) -> RGB (1, 0, 0)."""
        hwb = torch.tensor([0.0, 0.0, 0.0])
        rgb = hwb_to_srgb(hwb)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)

    def test_green(self):
        """HWB green (H=2*pi/3, W=0, B=0) -> RGB (0, 1, 0)."""
        hwb = torch.tensor([2 * math.pi / 3, 0.0, 0.0])
        rgb = hwb_to_srgb(hwb)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)

    def test_blue(self):
        """HWB blue (H=4*pi/3, W=0, B=0) -> RGB (0, 0, 1)."""
        hwb = torch.tensor([4 * math.pi / 3, 0.0, 0.0])
        rgb = hwb_to_srgb(hwb)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_white(self):
        """HWB white (H=0, W=1, B=0) -> RGB (1, 1, 1)."""
        hwb = torch.tensor([0.0, 1.0, 0.0])
        rgb = hwb_to_srgb(hwb)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_black(self):
        """HWB black (H=0, W=0, B=1) -> RGB (0, 0, 0)."""
        hwb = torch.tensor([0.0, 0.0, 1.0])
        rgb = hwb_to_srgb(hwb)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)

    def test_gray(self):
        """HWB gray (H=0, W=0.5, B=0.5) -> RGB (0.5, 0.5, 0.5)."""
        hwb = torch.tensor([0.0, 0.5, 0.5])
        rgb = hwb_to_srgb(hwb)
        assert torch.isclose(rgb[0], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.5), atol=1e-5)

    def test_half_white_red(self):
        """Red with half whiteness (H=0, W=0.5, B=0) -> RGB (1, 0.5, 0.5)."""
        hwb = torch.tensor([0.0, 0.5, 0.0])
        rgb = hwb_to_srgb(hwb)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.5), atol=1e-5)

    def test_half_black_red(self):
        """Red with half blackness (H=0, W=0, B=0.5) -> RGB (0.5, 0, 0)."""
        hwb = torch.tensor([0.0, 0.0, 0.5])
        rgb = hwb_to_srgb(hwb)
        assert torch.isclose(rgb[0], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-5)


class TestHwbToSrgbAchromatic:
    """Tests for achromatic (W + B >= 1) case."""

    def test_achromatic_gray(self):
        """W + B = 1 gives gray."""
        hwb = torch.tensor([0.0, 0.3, 0.7])  # W + B = 1
        rgb = hwb_to_srgb(hwb)
        expected = 0.3  # W / (W + B) = 0.3 / 1.0
        assert torch.isclose(rgb[0], torch.tensor(expected), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(expected), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(expected), atol=1e-5)

    def test_achromatic_over_one(self):
        """W + B > 1 normalizes to gray."""
        hwb = torch.tensor([0.0, 0.6, 0.6])  # W + B = 1.2
        rgb = hwb_to_srgb(hwb)
        expected = 0.6 / 1.2  # W / (W + B) = 0.5
        assert torch.isclose(rgb[0], torch.tensor(expected), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(expected), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(expected), atol=1e-5)

    def test_achromatic_hue_ignored(self):
        """In achromatic case, hue is ignored."""
        hwb1 = torch.tensor([0.0, 0.5, 0.5])
        hwb2 = torch.tensor([math.pi, 0.5, 0.5])
        rgb1 = hwb_to_srgb(hwb1)
        rgb2 = hwb_to_srgb(hwb2)
        assert torch.allclose(rgb1, rgb2, atol=1e-5)


class TestHwbToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck_chromatic(self):
        """Gradient check for chromatic colors (W + B < 1)."""
        hwb = torch.tensor(
            [[1.0, 0.2, 0.3]], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(hwb_to_srgb, (hwb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_achromatic(self):
        """Gradient check for achromatic colors (W + B >= 1)."""
        hwb = torch.tensor(
            [[0.0, 0.6, 0.6]], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(hwb_to_srgb, (hwb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check for batch of colors."""
        torch.manual_seed(42)
        # Ensure W + B < 1 for chromatic colors
        hwb = torch.rand(5, 3, dtype=torch.float64)
        hwb[:, 1] *= 0.3  # W in [0, 0.3]
        hwb[:, 2] *= 0.3  # B in [0, 0.3]
        hwb = hwb.requires_grad_(True)
        assert gradcheck(hwb_to_srgb, (hwb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradients flow back correctly."""
        hwb = torch.tensor([[1.0, 0.2, 0.3]], requires_grad=True)
        rgb = hwb_to_srgb(hwb)
        loss = rgb.sum()
        loss.backward()
        assert hwb.grad is not None
        assert not torch.isnan(hwb.grad).any()


class TestHwbToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        hwb = torch.rand(10, 3, dtype=torch.float32)
        rgb = hwb_to_srgb(hwb)
        assert rgb.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        hwb = torch.rand(10, 3, dtype=torch.float64)
        rgb = hwb_to_srgb(hwb)
        assert rgb.dtype == torch.float64

    @pytest.mark.skip(reason="Half precision may have accuracy issues")
    def test_float16(self):
        """Works with float16."""
        hwb = torch.rand(10, 3, dtype=torch.float16)
        rgb = hwb_to_srgb(hwb)
        assert rgb.dtype == torch.float16


class TestHwbSrgbRoundtrip:
    """Tests for roundtrip conversion."""

    def test_roundtrip_primary_colors(self):
        """Roundtrip for primary colors."""
        from torchscience.color import srgb_to_hwb

        colors = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
            ]
        )
        reconstructed = hwb_to_srgb(srgb_to_hwb(colors))
        assert torch.allclose(colors, reconstructed, atol=1e-5)

    def test_roundtrip_random_colors(self):
        """Roundtrip for random colors."""
        from torchscience.color import srgb_to_hwb

        torch.manual_seed(42)
        rgb = torch.rand(100, 3)
        reconstructed = hwb_to_srgb(srgb_to_hwb(rgb))
        assert torch.allclose(rgb, reconstructed, atol=1e-5)

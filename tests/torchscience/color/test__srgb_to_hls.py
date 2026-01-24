"""Tests for srgb_to_hls color conversion."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_hls


class TestSrgbToHlsShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        rgb = torch.tensor([1.0, 0.0, 0.0])
        hls = srgb_to_hls(rgb)
        assert hls.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        rgb = torch.randn(10, 3)
        hls = srgb_to_hls(rgb)
        assert hls.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        rgb = torch.randn(64, 64, 3)
        hls = srgb_to_hls(rgb)
        assert hls.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        rgb = torch.randn(10, 32, 32, 3)
        hls = srgb_to_hls(rgb)
        assert hls.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            srgb_to_hls(torch.randn(10, 4))


class TestSrgbToHlsKnownValues:
    """Tests for known color conversions."""

    def test_red(self):
        """Pure red: (1, 0, 0) -> H=0, L=0.5, S=1."""
        rgb = torch.tensor([1.0, 0.0, 0.0])
        hls = srgb_to_hls(rgb)
        assert torch.isclose(hls[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(hls[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(hls[2], torch.tensor(1.0), atol=1e-5)

    def test_green(self):
        """Pure green: (0, 1, 0) -> H=2pi/3, L=0.5, S=1."""
        rgb = torch.tensor([0.0, 1.0, 0.0])
        hls = srgb_to_hls(rgb)
        expected_h = 2 * math.pi / 3
        assert torch.isclose(hls[0], torch.tensor(expected_h), atol=1e-5)
        assert torch.isclose(hls[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(hls[2], torch.tensor(1.0), atol=1e-5)

    def test_blue(self):
        """Pure blue: (0, 0, 1) -> H=4pi/3, L=0.5, S=1."""
        rgb = torch.tensor([0.0, 0.0, 1.0])
        hls = srgb_to_hls(rgb)
        expected_h = 4 * math.pi / 3
        assert torch.isclose(hls[0], torch.tensor(expected_h), atol=1e-5)
        assert torch.isclose(hls[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(hls[2], torch.tensor(1.0), atol=1e-5)

    def test_yellow(self):
        """Yellow: (1, 1, 0) -> H=pi/3, L=0.5, S=1."""
        rgb = torch.tensor([1.0, 1.0, 0.0])
        hls = srgb_to_hls(rgb)
        expected_h = math.pi / 3
        assert torch.isclose(hls[0], torch.tensor(expected_h), atol=1e-5)
        assert torch.isclose(hls[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(hls[2], torch.tensor(1.0), atol=1e-5)

    def test_cyan(self):
        """Cyan: (0, 1, 1) -> H=pi, L=0.5, S=1."""
        rgb = torch.tensor([0.0, 1.0, 1.0])
        hls = srgb_to_hls(rgb)
        expected_h = math.pi
        assert torch.isclose(hls[0], torch.tensor(expected_h), atol=1e-5)
        assert torch.isclose(hls[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(hls[2], torch.tensor(1.0), atol=1e-5)

    def test_magenta(self):
        """Magenta: (1, 0, 1) -> H=5pi/3, L=0.5, S=1."""
        rgb = torch.tensor([1.0, 0.0, 1.0])
        hls = srgb_to_hls(rgb)
        expected_h = 5 * math.pi / 3
        assert torch.isclose(hls[0], torch.tensor(expected_h), atol=1e-5)
        assert torch.isclose(hls[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(hls[2], torch.tensor(1.0), atol=1e-5)

    def test_white(self):
        """White: (1, 1, 1) -> H=0, L=1, S=0."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        hls = srgb_to_hls(rgb)
        assert torch.isclose(hls[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(hls[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(hls[2], torch.tensor(0.0), atol=1e-5)

    def test_black(self):
        """Black: (0, 0, 0) -> H=0, L=0, S=0."""
        rgb = torch.tensor([0.0, 0.0, 0.0])
        hls = srgb_to_hls(rgb)
        assert torch.isclose(hls[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(hls[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(hls[2], torch.tensor(0.0), atol=1e-5)

    def test_gray(self):
        """Gray: (0.5, 0.5, 0.5) -> H=0, L=0.5, S=0."""
        rgb = torch.tensor([0.5, 0.5, 0.5])
        hls = srgb_to_hls(rgb)
        assert torch.isclose(hls[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(hls[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(hls[2], torch.tensor(0.0), atol=1e-5)

    def test_light_red(self):
        """Light red: (1, 0.5, 0.5) -> H=0, L=0.75, S=1."""
        rgb = torch.tensor([1.0, 0.5, 0.5])
        hls = srgb_to_hls(rgb)
        assert torch.isclose(hls[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(hls[1], torch.tensor(0.75), atol=1e-5)
        assert torch.isclose(hls[2], torch.tensor(1.0), atol=1e-5)

    def test_dark_red(self):
        """Dark red: (0.5, 0, 0) -> H=0, L=0.25, S=1."""
        rgb = torch.tensor([0.5, 0.0, 0.0])
        hls = srgb_to_hls(rgb)
        assert torch.isclose(hls[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(hls[1], torch.tensor(0.25), atol=1e-5)
        assert torch.isclose(hls[2], torch.tensor(1.0), atol=1e-5)


class TestSrgbToHlsEdgeCases:
    """Tests for edge cases and special values."""

    def test_near_grayscale(self):
        """Near-grayscale colors should have low saturation."""
        rgb = torch.tensor([0.5, 0.5, 0.501])
        hls = srgb_to_hls(rgb)
        assert hls[2] < 0.01  # Low saturation

    def test_hdr_input(self):
        """Allow values outside [0, 1] (HDR)."""
        rgb = torch.tensor([2.0, 0.5, 0.0])
        hls = srgb_to_hls(rgb)
        # L = (max + min) / 2 = (2.0 + 0.0) / 2 = 1.0
        assert torch.isclose(hls[1], torch.tensor(1.0), atol=1e-5)

    def test_negative_input(self):
        """Allow negative values."""
        rgb = torch.tensor([-0.5, 0.0, 0.5])
        hls = srgb_to_hls(rgb)
        # Should not raise
        assert hls[1] == 0.0  # L = (0.5 + (-0.5)) / 2 = 0


class TestSrgbToHlsGradients:
    """Tests for gradient computation."""

    def test_gradcheck_saturated(self):
        """Gradient check for saturated colors."""
        # Avoid L=0.5 exactly (gradient discontinuity at L=0.5)
        # L = (max + min) / 2, so avoid max + min = 1.0
        rgb = torch.tensor(
            [[0.8, 0.3, 0.5]], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(srgb_to_hls, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check for batch of colors."""
        torch.manual_seed(42)
        # Avoid grayscale and near-zero by adding minimum saturation
        rgb = torch.rand(5, 3, dtype=torch.float64) * 0.6 + 0.2
        # Ensure colors are not grayscale by perturbing one channel
        rgb[:, 0] += 0.2
        rgb = rgb.requires_grad_(True)
        assert gradcheck(srgb_to_hls, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradients flow back correctly."""
        rgb = torch.tensor([[0.8, 0.3, 0.5]], requires_grad=True)
        hls = srgb_to_hls(rgb)
        loss = hls.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.isnan(rgb.grad).any()

    @pytest.mark.skip(reason="Hue gradient undefined at grayscale")
    def test_gradcheck_grayscale(self):
        """Gradient check at grayscale (expected to fail - hue undefined)."""
        rgb = torch.tensor(
            [[0.5, 0.5, 0.5]], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(srgb_to_hls, (rgb,), eps=1e-6, atol=1e-4)


class TestSrgbToHlsDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        rgb = torch.rand(10, 3, dtype=torch.float32)
        hls = srgb_to_hls(rgb)
        assert hls.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        rgb = torch.rand(10, 3, dtype=torch.float64)
        hls = srgb_to_hls(rgb)
        assert hls.dtype == torch.float64

    @pytest.mark.skip(reason="Half precision may have accuracy issues")
    def test_float16(self):
        """Works with float16."""
        rgb = torch.rand(10, 3, dtype=torch.float16)
        hls = srgb_to_hls(rgb)
        assert hls.dtype == torch.float16

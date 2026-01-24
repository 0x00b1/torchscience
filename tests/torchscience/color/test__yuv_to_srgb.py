"""Tests for yuv_to_srgb color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_yuv, yuv_to_srgb


class TestYuvToSrgbShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        yuv = torch.tensor([0.5, 0.0, 0.0])
        rgb = yuv_to_srgb(yuv)
        assert rgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        yuv = torch.rand(10, 3)
        rgb = yuv_to_srgb(yuv)
        assert rgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        yuv = torch.rand(64, 64, 3)
        rgb = yuv_to_srgb(yuv)
        assert rgb.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        yuv = torch.rand(10, 32, 32, 3)
        rgb = yuv_to_srgb(yuv)
        assert rgb.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            yuv_to_srgb(torch.randn(10, 4))


class TestYuvToSrgbKnownValues:
    """Tests for known color conversions (BT.601 reference values)."""

    def test_white(self):
        """White: (1, 0, 0) -> (1, 1, 1)."""
        yuv = torch.tensor([1.0, 0.0, 0.0])
        rgb = yuv_to_srgb(yuv)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0, 0)."""
        yuv = torch.tensor([0.0, 0.0, 0.0])
        rgb = yuv_to_srgb(yuv)
        assert torch.allclose(rgb, torch.zeros(3), atol=1e-7)

    def test_red(self):
        """Pure red: (0.299, -0.1471376975, 0.615) -> (1, 0, 0)."""
        yuv = torch.tensor([0.299, -0.1471376975, 0.615])
        rgb = yuv_to_srgb(yuv)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-4)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-4)

    def test_green(self):
        """Pure green: (0.587, -0.2888623025, -0.5149857347) -> (0, 1, 0)."""
        yuv = torch.tensor([0.587, -0.2888623025, -0.5149857347])
        rgb = yuv_to_srgb(yuv)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-4)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-4)

    def test_blue(self):
        """Pure blue: (0.114, 0.436, -0.1000142653) -> (0, 0, 1)."""
        yuv = torch.tensor([0.114, 0.436, -0.1000142653])
        rgb = yuv_to_srgb(yuv)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-4)

    def test_mid_gray(self):
        """Mid gray (0.5, 0, 0)."""
        yuv = torch.tensor([0.5, 0.0, 0.0])
        rgb = yuv_to_srgb(yuv)
        assert torch.isclose(rgb[0], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(0.5), atol=1e-5)


class TestYuvToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        yuv = torch.rand(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(yuv_to_srgb, (yuv,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        yuv = torch.rand(4, 4, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(yuv_to_srgb, (yuv,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        yuv = torch.tensor([[0.5, 0.0, 0.0]], requires_grad=True)
        rgb = yuv_to_srgb(yuv)
        loss = rgb.sum()
        loss.backward()
        assert yuv.grad is not None
        assert not torch.any(torch.isnan(yuv.grad))


class TestYuvToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        yuv = torch.rand(10, 3, dtype=torch.float32)
        rgb = yuv_to_srgb(yuv)
        assert rgb.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        yuv = torch.rand(10, 3, dtype=torch.float64)
        rgb = yuv_to_srgb(yuv)
        assert rgb.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        yuv = torch.rand(10, 3, dtype=torch.bfloat16)
        rgb = yuv_to_srgb(yuv)
        assert rgb.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        yuv = torch.rand(10, 3, dtype=torch.float16)
        rgb = yuv_to_srgb(yuv)
        assert rgb.dtype == torch.float16


class TestYuvToSrgbEdgeCases:
    """Tests for edge cases."""

    def test_empty_tensor(self):
        """Test with empty tensor."""
        yuv = torch.empty(0, 3)
        rgb = yuv_to_srgb(yuv)
        assert rgb.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        yuv = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not yuv.is_contiguous()
        rgb = yuv_to_srgb(yuv)
        assert rgb.shape == yuv.shape


class TestRoundTrip:
    """Tests for round-trip conversion sRGB -> YUV -> sRGB."""

    def test_round_trip_random(self):
        """Random values should round-trip correctly."""
        rgb_original = torch.rand(100, 3, dtype=torch.float64)
        yuv = srgb_to_yuv(rgb_original)
        rgb_recovered = yuv_to_srgb(yuv)
        assert torch.allclose(rgb_original, rgb_recovered, atol=1e-5)

    def test_round_trip_primaries(self):
        """Primary colors should round-trip correctly."""
        primaries = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 1.0],  # White
                [0.0, 0.0, 0.0],  # Black
                [0.5, 0.5, 0.5],  # Gray
            ],
            dtype=torch.float64,
        )
        yuv = srgb_to_yuv(primaries)
        rgb_recovered = yuv_to_srgb(yuv)
        assert torch.allclose(primaries, rgb_recovered, atol=1e-5)

    def test_round_trip_batch(self):
        """Batch round-trip conversion."""
        rgb_original = torch.rand(8, 16, 16, 3, dtype=torch.float64)
        yuv = srgb_to_yuv(rgb_original)
        rgb_recovered = yuv_to_srgb(yuv)
        assert torch.allclose(rgb_original, rgb_recovered, atol=1e-5)

    def test_round_trip_gradient(self):
        """Gradient should flow through round-trip."""
        rgb = torch.rand(5, 3, dtype=torch.float64, requires_grad=True)

        def round_trip(x):
            return yuv_to_srgb(srgb_to_yuv(x))

        assert gradcheck(round_trip, (rgb,), eps=1e-6, atol=1e-4)

"""Tests for xyz_to_srgb color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_xyz, xyz_to_srgb


class TestXyzToSrgbShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        xyz = torch.tensor([0.9505, 1.0, 1.089])
        rgb = xyz_to_srgb(xyz)
        assert rgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        xyz = torch.randn(10, 3).abs()
        rgb = xyz_to_srgb(xyz)
        assert rgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        xyz = torch.randn(64, 64, 3).abs()
        rgb = xyz_to_srgb(xyz)
        assert rgb.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        xyz = torch.randn(10, 32, 32, 3).abs()
        rgb = xyz_to_srgb(xyz)
        assert rgb.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            xyz_to_srgb(torch.randn(10, 4))


class TestXyzToSrgbKnownValues:
    """Tests for known color conversions."""

    def test_white_d65(self):
        """D65 white point: (0.9505, 1.0, 1.089) -> (1, 1, 1)."""
        xyz = torch.tensor([0.9504559, 1.0, 1.0890578])
        rgb = xyz_to_srgb(xyz)
        assert torch.allclose(rgb, torch.ones(3), atol=1e-4)

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0, 0)."""
        xyz = torch.tensor([0.0, 0.0, 0.0])
        rgb = xyz_to_srgb(xyz)
        assert torch.allclose(rgb, torch.zeros(3), atol=1e-7)

    def test_red_primary(self):
        """sRGB red primary XYZ -> (1, 0, 0)."""
        xyz = torch.tensor([0.4124564, 0.2126729, 0.0193339])
        rgb = xyz_to_srgb(xyz)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-4)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-4)

    def test_green_primary(self):
        """sRGB green primary XYZ -> (0, 1, 0)."""
        xyz = torch.tensor([0.3575761, 0.7151522, 0.1191920])
        rgb = xyz_to_srgb(xyz)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-4)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-4)

    def test_blue_primary(self):
        """sRGB blue primary XYZ -> (0, 0, 1)."""
        xyz = torch.tensor([0.1804375, 0.0721750, 0.9503041])
        rgb = xyz_to_srgb(xyz)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-4)


class TestXyzToSrgbRoundTrip:
    """Tests for round-trip conversion."""

    def test_round_trip_primaries(self):
        """RGB -> XYZ -> RGB should recover original for primaries."""
        rgb_original = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
            ]
        )
        xyz = srgb_to_xyz(rgb_original)
        rgb_recovered = xyz_to_srgb(xyz)
        assert torch.allclose(rgb_original, rgb_recovered, atol=1e-5)

    def test_round_trip_random(self):
        """RGB -> XYZ -> RGB should recover original for random colors."""
        rgb_original = torch.rand(100, 3)
        xyz = srgb_to_xyz(rgb_original)
        rgb_recovered = xyz_to_srgb(xyz)
        assert torch.allclose(rgb_original, rgb_recovered, atol=1e-5)

    def test_round_trip_batch(self):
        """Round trip on batched image."""
        rgb_original = torch.rand(8, 8, 3)
        xyz = srgb_to_xyz(rgb_original)
        rgb_recovered = xyz_to_srgb(xyz)
        assert torch.allclose(rgb_original, rgb_recovered, atol=1e-5)


class TestXyzToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        xyz = (
            torch.rand(5, 3, dtype=torch.float64, requires_grad=True) * 0.5
            + 0.1
        )
        assert gradcheck(xyz_to_srgb, (xyz,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        xyz = (
            torch.rand(4, 4, 3, dtype=torch.float64, requires_grad=True) * 0.5
            + 0.1
        )
        assert gradcheck(xyz_to_srgb, (xyz,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        xyz = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)
        rgb = xyz_to_srgb(xyz)
        loss = rgb.sum()
        loss.backward()
        assert xyz.grad is not None
        assert not torch.any(torch.isnan(xyz.grad))


class TestXyzToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        xyz = torch.rand(10, 3, dtype=torch.float32)
        rgb = xyz_to_srgb(xyz)
        assert rgb.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        xyz = torch.rand(10, 3, dtype=torch.float64)
        rgb = xyz_to_srgb(xyz)
        assert rgb.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        xyz = torch.rand(10, 3, dtype=torch.bfloat16)
        rgb = xyz_to_srgb(xyz)
        assert rgb.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        xyz = torch.rand(10, 3, dtype=torch.float16)
        rgb = xyz_to_srgb(xyz)
        assert rgb.dtype == torch.float16


class TestXyzToSrgbEdgeCases:
    """Tests for edge cases."""

    def test_out_of_gamut(self):
        """Test with out-of-gamut XYZ (produces RGB outside [0,1])."""
        xyz = torch.tensor([0.5, 0.1, 0.9])  # Out of sRGB gamut
        rgb = xyz_to_srgb(xyz)
        assert not torch.any(torch.isnan(rgb))

    def test_negative_xyz(self):
        """Test with negative XYZ values."""
        xyz = torch.tensor([-0.1, 0.5, 0.5])
        rgb = xyz_to_srgb(xyz)
        assert not torch.any(torch.isnan(rgb))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        xyz = torch.empty(0, 3)
        rgb = xyz_to_srgb(xyz)
        assert rgb.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        xyz = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not xyz.is_contiguous()
        rgb = xyz_to_srgb(xyz)
        assert rgb.shape == xyz.shape

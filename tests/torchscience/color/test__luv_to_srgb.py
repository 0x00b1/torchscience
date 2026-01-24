"""Tests for luv_to_srgb color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import luv_to_srgb, srgb_to_luv


class TestLuvToSrgbShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        luv = torch.tensor([50.0, 0.0, 0.0])
        rgb = luv_to_srgb(luv)
        assert rgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        luv = torch.randn(10, 3) * 50 + torch.tensor([50.0, 0.0, 0.0])
        rgb = luv_to_srgb(luv)
        assert rgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        luv = torch.randn(64, 64, 3) * 50 + torch.tensor([50.0, 0.0, 0.0])
        rgb = luv_to_srgb(luv)
        assert rgb.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        luv = torch.randn(10, 32, 32, 3) * 50 + torch.tensor([50.0, 0.0, 0.0])
        rgb = luv_to_srgb(luv)
        assert rgb.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            luv_to_srgb(torch.randn(10, 4))


class TestLuvToSrgbKnownValues:
    """Tests for known color conversions (reference: colour-science library)."""

    def test_white_d65(self):
        """D65 white point: (100, 0, 0) -> (1, 1, 1)."""
        luv = torch.tensor([100.0, 0.0, 0.0])
        rgb = luv_to_srgb(luv)
        # White in sRGB should be R=G=B=1
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-3)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-3)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-3)

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0, 0)."""
        luv = torch.tensor([0.0, 0.0, 0.0])
        rgb = luv_to_srgb(luv)
        # Black should have R=G=B=0
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-4)

    def test_red(self):
        """LUV red primary back to sRGB."""
        # Reference LUV values for sRGB red
        luv = torch.tensor([53.2329, 175.0150, 37.7564])
        rgb = luv_to_srgb(luv)
        # Should be close to pure red
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=0.02)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=0.02)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=0.02)

    def test_green(self):
        """LUV green primary back to sRGB."""
        # Reference LUV values for sRGB green
        luv = torch.tensor([87.7370, -83.0776, 107.3985])
        rgb = luv_to_srgb(luv)
        # Should be close to pure green
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=0.02)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=0.02)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=0.02)

    def test_blue(self):
        """LUV blue primary back to sRGB."""
        # Reference LUV values for sRGB blue
        luv = torch.tensor([32.3026, -9.4054, -130.3423])
        rgb = luv_to_srgb(luv)
        # Should be close to pure blue
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=0.02)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=0.02)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=0.02)

    def test_mid_gray(self):
        """Mid gray in LUV (u=0, v=0) should give R=G=B."""
        luv = torch.tensor([53.39, 0.0, 0.0])  # Approx sRGB (0.5, 0.5, 0.5)
        rgb = luv_to_srgb(luv)
        # Gray should have R=G=B
        assert torch.isclose(rgb[0], rgb[1], atol=1e-3)
        assert torch.isclose(rgb[1], rgb[2], atol=1e-3)


class TestLuvToSrgbRoundTrip:
    """Tests for round-trip conversion accuracy."""

    def test_round_trip_srgb_to_luv_to_srgb(self):
        """sRGB -> LUV -> sRGB should be identity."""
        # Test with values in valid sRGB gamut
        rgb_original = torch.rand(100, 3)
        luv = srgb_to_luv(rgb_original)
        rgb_recovered = luv_to_srgb(luv)
        assert torch.allclose(rgb_original, rgb_recovered, atol=1e-4)

    def test_round_trip_luv_to_srgb_to_luv(self):
        """LUV -> sRGB -> LUV should be identity for in-gamut colors."""
        # Generate LUV values that are likely in sRGB gamut
        # L in [10, 90], u in [-50, 50], v in [-50, 50]
        luv_original = torch.zeros(100, 3)
        luv_original[:, 0] = torch.rand(100) * 80 + 10  # L
        luv_original[:, 1] = torch.rand(100) * 100 - 50  # u
        luv_original[:, 2] = torch.rand(100) * 100 - 50  # v

        rgb = luv_to_srgb(luv_original)
        # Filter to only in-gamut colors
        mask = (rgb >= 0).all(dim=-1) & (rgb <= 1).all(dim=-1)
        if mask.sum() > 0:
            luv_recovered = srgb_to_luv(rgb[mask])
            assert torch.allclose(luv_original[mask], luv_recovered, atol=1e-3)

    def test_round_trip_primaries(self):
        """Round trip for sRGB primaries."""
        primaries = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 1.0],  # White
                [0.0, 0.0, 0.0],  # Black
                [0.5, 0.5, 0.5],  # Gray
            ]
        )
        luv = srgb_to_luv(primaries)
        rgb_recovered = luv_to_srgb(luv)
        assert torch.allclose(primaries, rgb_recovered, atol=1e-4)


class TestLuvToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        # Use in-gamut LUV values
        luv = torch.zeros(5, 3, dtype=torch.float64, requires_grad=True)
        luv.data[:, 0] = torch.rand(5, dtype=torch.float64) * 80 + 10  # L
        luv.data[:, 1] = torch.rand(5, dtype=torch.float64) * 40 - 20  # u
        luv.data[:, 2] = torch.rand(5, dtype=torch.float64) * 40 - 20  # v
        assert gradcheck(luv_to_srgb, (luv,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        luv = torch.zeros(4, 4, 3, dtype=torch.float64, requires_grad=True)
        luv.data[..., 0] = torch.rand(4, 4, dtype=torch.float64) * 80 + 10
        luv.data[..., 1] = torch.rand(4, 4, dtype=torch.float64) * 40 - 20
        luv.data[..., 2] = torch.rand(4, 4, dtype=torch.float64) * 40 - 20
        assert gradcheck(luv_to_srgb, (luv,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        luv = torch.tensor([[50.0, 10.0, -10.0]], requires_grad=True)
        rgb = luv_to_srgb(luv)
        loss = rgb.sum()
        loss.backward()
        assert luv.grad is not None
        assert not torch.any(torch.isnan(luv.grad))


class TestLuvToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        luv = torch.tensor([[50.0, 0.0, 0.0]], dtype=torch.float32)
        rgb = luv_to_srgb(luv)
        assert rgb.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        luv = torch.tensor([[50.0, 0.0, 0.0]], dtype=torch.float64)
        rgb = luv_to_srgb(luv)
        assert rgb.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        luv = torch.tensor([[50.0, 0.0, 0.0]], dtype=torch.bfloat16)
        rgb = luv_to_srgb(luv)
        assert rgb.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        luv = torch.tensor([[50.0, 0.0, 0.0]], dtype=torch.float16)
        rgb = luv_to_srgb(luv)
        assert rgb.dtype == torch.float16


class TestLuvToSrgbEdgeCases:
    """Tests for edge cases."""

    def test_hdr_values(self):
        """Test with HDR LUV values (L > 100)."""
        luv = torch.tensor([120.0, 0.0, 0.0])
        rgb = luv_to_srgb(luv)
        assert not torch.any(torch.isnan(rgb))
        # RGB values should exceed 1 for super-white
        assert rgb[0] > 1

    def test_negative_lightness(self):
        """Test with negative L (out of spec)."""
        luv = torch.tensor([-10.0, 0.0, 0.0])
        rgb = luv_to_srgb(luv)
        assert not torch.any(torch.isnan(rgb))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        luv = torch.empty(0, 3)
        rgb = luv_to_srgb(luv)
        assert rgb.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        luv = torch.rand(10, 10, 3) * 100
        luv = luv[:, ::2, :]  # Non-contiguous
        assert not luv.is_contiguous()
        rgb = luv_to_srgb(luv)
        assert rgb.shape == luv.shape

    def test_extreme_chroma(self):
        """Test with extreme u* and v* values."""
        luv = torch.tensor([50.0, 150.0, -150.0])
        rgb = luv_to_srgb(luv)
        assert not torch.any(torch.isnan(rgb))
        # This is out of gamut, so RGB may have values outside [0,1]

    def test_l_equals_zero(self):
        """Test with L=0 (black point)."""
        luv = torch.tensor([0.0, 50.0, 50.0])
        rgb = luv_to_srgb(luv)
        assert not torch.any(torch.isnan(rgb))
        # Should still be black regardless of u* and v*
        assert torch.allclose(rgb, torch.zeros(3), atol=1e-4)

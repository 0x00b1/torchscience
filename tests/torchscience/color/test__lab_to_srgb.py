"""Tests for lab_to_srgb color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import lab_to_srgb, srgb_to_lab


class TestLabToSrgbShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        lab = torch.tensor([50.0, 0.0, 0.0])
        rgb = lab_to_srgb(lab)
        assert rgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        lab = torch.randn(10, 3) * 50 + torch.tensor([50.0, 0.0, 0.0])
        rgb = lab_to_srgb(lab)
        assert rgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        lab = torch.randn(64, 64, 3) * 50 + torch.tensor([50.0, 0.0, 0.0])
        rgb = lab_to_srgb(lab)
        assert rgb.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        lab = torch.randn(10, 32, 32, 3) * 50 + torch.tensor([50.0, 0.0, 0.0])
        rgb = lab_to_srgb(lab)
        assert rgb.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            lab_to_srgb(torch.randn(10, 4))


class TestLabToSrgbKnownValues:
    """Tests for known color conversions (reference: colour-science library)."""

    def test_white_d65(self):
        """D65 white point: (100, 0, 0) -> (1, 1, 1)."""
        lab = torch.tensor([100.0, 0.0, 0.0])
        rgb = lab_to_srgb(lab)
        # White in sRGB should be R=G=B=1
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-3)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-3)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-3)

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0, 0)."""
        lab = torch.tensor([0.0, 0.0, 0.0])
        rgb = lab_to_srgb(lab)
        # Black should have R=G=B=0
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-4)

    def test_red(self):
        """Lab red primary back to sRGB."""
        # Reference Lab values for sRGB red
        lab = torch.tensor([53.2329, 80.1093, 67.2201])
        rgb = lab_to_srgb(lab)
        # Should be close to pure red
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=0.01)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=0.01)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=0.01)

    def test_green(self):
        """Lab green primary back to sRGB."""
        # Reference Lab values for sRGB green
        lab = torch.tensor([87.7370, -86.1846, 83.1812])
        rgb = lab_to_srgb(lab)
        # Should be close to pure green
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=0.01)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=0.01)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=0.01)

    def test_blue(self):
        """Lab blue primary back to sRGB."""
        # Reference Lab values for sRGB blue
        lab = torch.tensor([32.3026, 79.1967, -107.8637])
        rgb = lab_to_srgb(lab)
        # Should be close to pure blue
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=0.01)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=0.01)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=0.01)

    def test_mid_gray(self):
        """Mid gray in Lab (a=0, b=0) should give R=G=B."""
        lab = torch.tensor([53.39, 0.0, 0.0])  # Approx sRGB (0.5, 0.5, 0.5)
        rgb = lab_to_srgb(lab)
        # Gray should have R=G=B
        assert torch.isclose(rgb[0], rgb[1], atol=1e-3)
        assert torch.isclose(rgb[1], rgb[2], atol=1e-3)


class TestLabToSrgbRoundTrip:
    """Tests for round-trip conversion accuracy."""

    def test_round_trip_srgb_to_lab_to_srgb(self):
        """sRGB -> Lab -> sRGB should be identity."""
        # Test with values in valid sRGB gamut
        rgb_original = torch.rand(100, 3)
        lab = srgb_to_lab(rgb_original)
        rgb_recovered = lab_to_srgb(lab)
        assert torch.allclose(rgb_original, rgb_recovered, atol=1e-4)

    def test_round_trip_lab_to_srgb_to_lab(self):
        """Lab -> sRGB -> Lab should be identity for in-gamut colors."""
        # Generate Lab values that are likely in sRGB gamut
        # L in [10, 90], a in [-50, 50], b in [-50, 50]
        lab_original = torch.zeros(100, 3)
        lab_original[:, 0] = torch.rand(100) * 80 + 10  # L
        lab_original[:, 1] = torch.rand(100) * 100 - 50  # a
        lab_original[:, 2] = torch.rand(100) * 100 - 50  # b

        rgb = lab_to_srgb(lab_original)
        # Filter to only in-gamut colors
        mask = (rgb >= 0).all(dim=-1) & (rgb <= 1).all(dim=-1)
        if mask.sum() > 0:
            lab_recovered = srgb_to_lab(rgb[mask])
            assert torch.allclose(lab_original[mask], lab_recovered, atol=1e-3)

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
        lab = srgb_to_lab(primaries)
        rgb_recovered = lab_to_srgb(lab)
        assert torch.allclose(primaries, rgb_recovered, atol=1e-4)


class TestLabToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        # Use in-gamut Lab values
        lab = torch.zeros(5, 3, dtype=torch.float64, requires_grad=True)
        lab.data[:, 0] = torch.rand(5, dtype=torch.float64) * 80 + 10  # L
        lab.data[:, 1] = torch.rand(5, dtype=torch.float64) * 40 - 20  # a
        lab.data[:, 2] = torch.rand(5, dtype=torch.float64) * 40 - 20  # b
        assert gradcheck(lab_to_srgb, (lab,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        lab = torch.zeros(4, 4, 3, dtype=torch.float64, requires_grad=True)
        lab.data[..., 0] = torch.rand(4, 4, dtype=torch.float64) * 80 + 10
        lab.data[..., 1] = torch.rand(4, 4, dtype=torch.float64) * 40 - 20
        lab.data[..., 2] = torch.rand(4, 4, dtype=torch.float64) * 40 - 20
        assert gradcheck(lab_to_srgb, (lab,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        lab = torch.tensor([[50.0, 10.0, -10.0]], requires_grad=True)
        rgb = lab_to_srgb(lab)
        loss = rgb.sum()
        loss.backward()
        assert lab.grad is not None
        assert not torch.any(torch.isnan(lab.grad))


class TestLabToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        lab = torch.tensor([[50.0, 0.0, 0.0]], dtype=torch.float32)
        rgb = lab_to_srgb(lab)
        assert rgb.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        lab = torch.tensor([[50.0, 0.0, 0.0]], dtype=torch.float64)
        rgb = lab_to_srgb(lab)
        assert rgb.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        lab = torch.tensor([[50.0, 0.0, 0.0]], dtype=torch.bfloat16)
        rgb = lab_to_srgb(lab)
        assert rgb.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        lab = torch.tensor([[50.0, 0.0, 0.0]], dtype=torch.float16)
        rgb = lab_to_srgb(lab)
        assert rgb.dtype == torch.float16


class TestLabToSrgbEdgeCases:
    """Tests for edge cases."""

    def test_hdr_values(self):
        """Test with HDR Lab values (L > 100)."""
        lab = torch.tensor([120.0, 0.0, 0.0])
        rgb = lab_to_srgb(lab)
        assert not torch.any(torch.isnan(rgb))
        # RGB values should exceed 1 for super-white
        assert rgb[0] > 1

    def test_negative_lightness(self):
        """Test with negative L (out of spec)."""
        lab = torch.tensor([-10.0, 0.0, 0.0])
        rgb = lab_to_srgb(lab)
        assert not torch.any(torch.isnan(rgb))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        lab = torch.empty(0, 3)
        rgb = lab_to_srgb(lab)
        assert rgb.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        lab = torch.rand(10, 10, 3) * 100
        lab = lab[:, ::2, :]  # Non-contiguous
        assert not lab.is_contiguous()
        rgb = lab_to_srgb(lab)
        assert rgb.shape == lab.shape

    def test_extreme_chroma(self):
        """Test with extreme a* and b* values."""
        lab = torch.tensor([50.0, 127.0, -127.0])
        rgb = lab_to_srgb(lab)
        assert not torch.any(torch.isnan(rgb))
        # This is out of gamut, so RGB may have values outside [0,1]

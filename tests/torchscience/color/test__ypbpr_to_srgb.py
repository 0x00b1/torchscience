"""Tests for ypbpr_to_srgb color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_ypbpr, ypbpr_to_srgb


class TestYpbprToSrgbShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        ypbpr = torch.tensor([0.5, 0.0, 0.0])
        rgb = ypbpr_to_srgb(ypbpr)
        assert rgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        ypbpr = torch.zeros(10, 3)
        ypbpr[..., 0] = torch.rand(10)  # Y in [0, 1]
        ypbpr[..., 1] = (torch.rand(10) - 0.5) * 0.5  # Pb in [-0.25, 0.25]
        ypbpr[..., 2] = (torch.rand(10) - 0.5) * 0.5  # Pr in [-0.25, 0.25]
        rgb = ypbpr_to_srgb(ypbpr)
        assert rgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        ypbpr = torch.zeros(64, 64, 3)
        ypbpr[..., 0] = 0.5
        rgb = ypbpr_to_srgb(ypbpr)
        assert rgb.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        ypbpr = torch.zeros(10, 32, 32, 3)
        ypbpr[..., 0] = 0.5
        rgb = ypbpr_to_srgb(ypbpr)
        assert rgb.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            ypbpr_to_srgb(torch.randn(10, 4))


class TestYpbprToSrgbKnownValues:
    """Tests for known color conversions (BT.601 inverse)."""

    def test_white(self):
        """White: Y=1, Pb=0, Pr=0 -> linear (1, 1, 1) -> sRGB (1, 1, 1)."""
        ypbpr = torch.tensor([1.0, 0.0, 0.0])
        rgb = ypbpr_to_srgb(ypbpr)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-5)

    def test_black(self):
        """Black: Y=0, Pb=0, Pr=0 -> (0, 0, 0)."""
        ypbpr = torch.tensor([0.0, 0.0, 0.0])
        rgb = ypbpr_to_srgb(ypbpr)
        assert torch.allclose(rgb, torch.zeros(3), atol=1e-7)

    def test_red(self):
        """Pure red: Y=0.299, Pb=-0.168736, Pr=0.5 -> (1, 0, 0)."""
        ypbpr = torch.tensor([0.299, -0.168736, 0.5])
        rgb = ypbpr_to_srgb(ypbpr)
        assert torch.isclose(rgb[0], torch.tensor(1.0), atol=1e-4)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-4)

    def test_green(self):
        """Pure green: Y=0.587, Pb=-0.331264, Pr=-0.418688 -> (0, 1, 0)."""
        ypbpr = torch.tensor([0.587, -0.331264, -0.418688])
        rgb = ypbpr_to_srgb(ypbpr)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[1], torch.tensor(1.0), atol=1e-4)
        assert torch.isclose(rgb[2], torch.tensor(0.0), atol=1e-4)

    def test_blue(self):
        """Pure blue: Y=0.114, Pb=0.5, Pr=-0.081312 -> (0, 0, 1)."""
        ypbpr = torch.tensor([0.114, 0.5, -0.081312])
        rgb = ypbpr_to_srgb(ypbpr)
        assert torch.isclose(rgb[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(rgb[2], torch.tensor(1.0), atol=1e-4)


class TestYpbprToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        # Use values that produce valid linear RGB
        ypbpr = torch.zeros(5, 3, dtype=torch.float64, requires_grad=True)
        ypbpr.data[..., 0] = torch.rand(5, dtype=torch.float64) * 0.5 + 0.25
        ypbpr.data[..., 1] = (torch.rand(5, dtype=torch.float64) - 0.5) * 0.2
        ypbpr.data[..., 2] = (torch.rand(5, dtype=torch.float64) - 0.5) * 0.2
        assert gradcheck(ypbpr_to_srgb, (ypbpr,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        ypbpr = torch.zeros(4, 4, 3, dtype=torch.float64, requires_grad=True)
        ypbpr.data[..., 0] = torch.rand(4, 4, dtype=torch.float64) * 0.5 + 0.25
        ypbpr.data[..., 1] = (
            torch.rand(4, 4, dtype=torch.float64) - 0.5
        ) * 0.2
        ypbpr.data[..., 2] = (
            torch.rand(4, 4, dtype=torch.float64) - 0.5
        ) * 0.2
        assert gradcheck(ypbpr_to_srgb, (ypbpr,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        ypbpr = torch.tensor([[0.5, 0.0, 0.0]], requires_grad=True)
        rgb = ypbpr_to_srgb(ypbpr)
        loss = rgb.sum()
        loss.backward()
        assert ypbpr.grad is not None
        assert not torch.any(torch.isnan(ypbpr.grad))


class TestYpbprToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        ypbpr = torch.zeros(10, 3, dtype=torch.float32)
        ypbpr[..., 0] = 0.5
        rgb = ypbpr_to_srgb(ypbpr)
        assert rgb.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        ypbpr = torch.zeros(10, 3, dtype=torch.float64)
        ypbpr[..., 0] = 0.5
        rgb = ypbpr_to_srgb(ypbpr)
        assert rgb.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        ypbpr = torch.zeros(10, 3, dtype=torch.bfloat16)
        ypbpr[..., 0] = 0.5
        rgb = ypbpr_to_srgb(ypbpr)
        assert rgb.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        ypbpr = torch.zeros(10, 3, dtype=torch.float16)
        ypbpr[..., 0] = 0.5
        rgb = ypbpr_to_srgb(ypbpr)
        assert rgb.dtype == torch.float16


class TestYpbprToSrgbEdgeCases:
    """Tests for edge cases."""

    def test_empty_tensor(self):
        """Test with empty tensor."""
        ypbpr = torch.empty(0, 3)
        rgb = ypbpr_to_srgb(ypbpr)
        assert rgb.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        ypbpr = torch.zeros(10, 10, 3)
        ypbpr[..., 0] = 0.5
        ypbpr = ypbpr[:, ::2, :]  # Non-contiguous
        assert not ypbpr.is_contiguous()
        rgb = ypbpr_to_srgb(ypbpr)
        assert rgb.shape == ypbpr.shape


class TestRoundTrip:
    """Tests for round-trip conversion sRGB -> YPbPr -> sRGB."""

    def test_round_trip_random(self):
        """Random values should round-trip correctly."""
        rgb_original = torch.rand(100, 3, dtype=torch.float64)
        ypbpr = srgb_to_ypbpr(rgb_original)
        rgb_recovered = ypbpr_to_srgb(ypbpr)
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
            ],
            dtype=torch.float64,
        )
        ypbpr = srgb_to_ypbpr(primaries)
        rgb_recovered = ypbpr_to_srgb(ypbpr)
        assert torch.allclose(primaries, rgb_recovered, atol=1e-5)

    def test_round_trip_batch(self):
        """Batch round-trip conversion."""
        rgb_original = torch.rand(8, 16, 16, 3, dtype=torch.float64)
        ypbpr = srgb_to_ypbpr(rgb_original)
        rgb_recovered = ypbpr_to_srgb(ypbpr)
        assert torch.allclose(rgb_original, rgb_recovered, atol=1e-5)

    def test_round_trip_gradient(self):
        """Gradient should flow through round-trip."""
        rgb = (
            torch.rand(5, 3, dtype=torch.float64, requires_grad=True) * 0.8
            + 0.1
        )

        def round_trip(x):
            return ypbpr_to_srgb(srgb_to_ypbpr(x))

        assert gradcheck(round_trip, (rgb,), eps=1e-6, atol=1e-4)


class TestYpbprVsYcbcr:
    """Tests to verify YPbPr differs from YCbCr correctly."""

    def test_different_from_ycbcr(self):
        """YPbPr should differ from YCbCr (no offset, linear RGB)."""
        from torchscience.color import srgb_to_ycbcr

        rgb = torch.tensor([[0.5, 0.5, 0.5]])
        ypbpr = srgb_to_ypbpr(rgb)
        ycbcr = srgb_to_ycbcr(rgb)

        # YCbCr has Cb, Cr centered at 0.5, YPbPr has Pb, Pr centered at 0
        # For gray, YCbCr Cb=Cr=0.5, YPbPr Pb=Pr=0
        assert torch.isclose(ycbcr[0, 1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(ycbcr[0, 2], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(ypbpr[0, 1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(ypbpr[0, 2], torch.tensor(0.0), atol=1e-5)

"""Tests for oklab_to_srgb color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import oklab_to_srgb, srgb_to_oklab


class TestOklabToSrgbShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        oklab = torch.tensor([1.0, 0.0, 0.0])
        rgb = oklab_to_srgb(oklab)
        assert rgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        oklab = torch.randn(10, 3)
        oklab[:, 0] = oklab[:, 0].abs()  # L should be positive
        rgb = oklab_to_srgb(oklab)
        assert rgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        oklab = torch.randn(64, 64, 3)
        oklab[:, :, 0] = oklab[:, :, 0].abs()
        rgb = oklab_to_srgb(oklab)
        assert rgb.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        oklab = torch.randn(10, 32, 32, 3)
        oklab[:, :, :, 0] = oklab[:, :, :, 0].abs()
        rgb = oklab_to_srgb(oklab)
        assert rgb.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            oklab_to_srgb(torch.randn(10, 4))


class TestOklabToSrgbKnownValues:
    """Tests for known color conversions."""

    def test_white(self):
        """Oklab white: (1, 0, 0) -> (1, 1, 1)."""
        oklab = torch.tensor([1.0, 0.0, 0.0])
        rgb = oklab_to_srgb(oklab)
        assert torch.allclose(rgb, torch.tensor([1.0, 1.0, 1.0]), atol=1e-4)

    def test_black(self):
        """Oklab black: (0, 0, 0) -> (0, 0, 0)."""
        oklab = torch.tensor([0.0, 0.0, 0.0])
        rgb = oklab_to_srgb(oklab)
        assert torch.allclose(rgb, torch.tensor([0.0, 0.0, 0.0]), atol=1e-4)

    def test_mid_gray(self):
        """Mid gray should convert back correctly."""
        # Convert sRGB mid gray to Oklab
        rgb_input = torch.tensor([0.5, 0.5, 0.5])
        oklab = srgb_to_oklab(rgb_input)
        # Convert back
        rgb_output = oklab_to_srgb(oklab)
        assert torch.allclose(rgb_input, rgb_output, atol=1e-4)


class TestOklabToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        # Use in-gamut Oklab values
        oklab = torch.rand(5, 3, dtype=torch.float64, requires_grad=True) * 0.5
        oklab = oklab.clone()
        oklab[:, 0] = oklab[:, 0] * 0.5 + 0.25  # L in [0.25, 0.75]
        oklab[:, 1:] = oklab[:, 1:] * 0.1 - 0.05  # a, b small
        oklab = oklab.detach().requires_grad_(True)
        assert gradcheck(oklab_to_srgb, (oklab,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        oklab = torch.rand(4, 4, 3, dtype=torch.float64) * 0.5
        oklab[:, :, 0] = oklab[:, :, 0] * 0.5 + 0.25
        oklab[:, :, 1:] = oklab[:, :, 1:] * 0.1 - 0.05
        oklab = oklab.clone().requires_grad_(True)
        assert gradcheck(oklab_to_srgb, (oklab,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        oklab = torch.tensor([[0.5, 0.05, -0.05]], requires_grad=True)
        rgb = oklab_to_srgb(oklab)
        loss = rgb.sum()
        loss.backward()
        assert oklab.grad is not None
        assert not torch.any(torch.isnan(oklab.grad))


class TestOklabToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        oklab = torch.rand(10, 3, dtype=torch.float32)
        rgb = oklab_to_srgb(oklab)
        assert rgb.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        oklab = torch.rand(10, 3, dtype=torch.float64)
        rgb = oklab_to_srgb(oklab)
        assert rgb.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        oklab = torch.rand(10, 3, dtype=torch.bfloat16)
        rgb = oklab_to_srgb(oklab)
        assert rgb.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        oklab = torch.rand(10, 3, dtype=torch.float16)
        rgb = oklab_to_srgb(oklab)
        assert rgb.dtype == torch.float16


class TestOklabToSrgbEdgeCases:
    """Tests for edge cases."""

    def test_out_of_gamut_values(self):
        """Test with out-of-gamut Oklab values."""
        oklab = torch.tensor([0.5, 0.5, 0.5])  # Out of gamut
        rgb = oklab_to_srgb(oklab)
        assert not torch.any(torch.isnan(rgb))
        # Some values may be outside [0, 1]

    def test_empty_tensor(self):
        """Test with empty tensor."""
        oklab = torch.empty(0, 3)
        rgb = oklab_to_srgb(oklab)
        assert rgb.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        oklab = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not oklab.is_contiguous()
        rgb = oklab_to_srgb(oklab)
        assert rgb.shape == oklab.shape


class TestOklabSrgbRoundTrip:
    """Tests for round-trip conversion."""

    def test_round_trip_in_gamut(self):
        """Round trip sRGB -> Oklab -> sRGB for in-gamut colors."""
        rgb_input = torch.rand(100, 3)
        oklab = srgb_to_oklab(rgb_input)
        rgb_output = oklab_to_srgb(oklab)
        # Use slightly relaxed tolerance for float32 precision
        assert torch.allclose(rgb_input, rgb_output, atol=1e-4)

    def test_round_trip_primary_colors(self):
        """Round trip for primary colors."""
        primaries = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
            ]
        )
        oklab = srgb_to_oklab(primaries)
        rgb_back = oklab_to_srgb(oklab)
        assert torch.allclose(primaries, rgb_back, atol=1e-5)

    def test_round_trip_grayscale(self):
        """Round trip for grayscale values."""
        grays = torch.linspace(0, 1, 11).unsqueeze(1).expand(-1, 3)
        oklab = srgb_to_oklab(grays)
        rgb_back = oklab_to_srgb(oklab)
        assert torch.allclose(grays, rgb_back, atol=1e-5)

    def test_round_trip_random_batch(self):
        """Round trip for random batch of colors."""
        rgb_input = torch.rand(64, 64, 3)
        oklab = srgb_to_oklab(rgb_input)
        rgb_output = oklab_to_srgb(oklab)
        # Use slightly relaxed tolerance for float32 precision
        assert torch.allclose(rgb_input, rgb_output, atol=1e-4)

    def test_round_trip_gradient_check(self):
        """Gradient check for round-trip conversion."""
        rgb = (
            torch.rand(5, 3, dtype=torch.float64, requires_grad=True) * 0.9
            + 0.05
        )

        def round_trip(x):
            return oklab_to_srgb(srgb_to_oklab(x))

        assert gradcheck(round_trip, (rgb,), eps=1e-6, atol=1e-4)

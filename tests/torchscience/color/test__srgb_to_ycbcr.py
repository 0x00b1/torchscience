"""Tests for srgb_to_ycbcr color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_ycbcr


class TestSrgbToYcbcrShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        ycbcr = srgb_to_ycbcr(rgb)
        assert ycbcr.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        rgb = torch.rand(10, 3)
        ycbcr = srgb_to_ycbcr(rgb)
        assert ycbcr.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        rgb = torch.rand(64, 64, 3)
        ycbcr = srgb_to_ycbcr(rgb)
        assert ycbcr.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        rgb = torch.rand(10, 32, 32, 3)
        ycbcr = srgb_to_ycbcr(rgb)
        assert ycbcr.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            srgb_to_ycbcr(torch.randn(10, 4))


class TestSrgbToYcbcrKnownValues:
    """Tests for known color conversions (BT.601 reference values)."""

    def test_white(self):
        """White: (1, 1, 1) -> (1, 0.5, 0.5)."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        ycbcr = srgb_to_ycbcr(rgb)
        # Y = 0.299 + 0.587 + 0.114 = 1.0
        # Cb = -0.168736 - 0.331264 + 0.5 + 0.5 = 0.5
        # Cr = 0.5 - 0.418688 - 0.081312 + 0.5 = 0.5
        assert torch.isclose(ycbcr[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(ycbcr[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(ycbcr[2], torch.tensor(0.5), atol=1e-5)

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0.5, 0.5)."""
        rgb = torch.tensor([0.0, 0.0, 0.0])
        ycbcr = srgb_to_ycbcr(rgb)
        assert torch.isclose(ycbcr[0], torch.tensor(0.0), atol=1e-7)
        assert torch.isclose(ycbcr[1], torch.tensor(0.5), atol=1e-7)
        assert torch.isclose(ycbcr[2], torch.tensor(0.5), atol=1e-7)

    def test_red(self):
        """Pure red: (1, 0, 0) -> (0.299, 0.331264, 1.0)."""
        rgb = torch.tensor([1.0, 0.0, 0.0])
        ycbcr = srgb_to_ycbcr(rgb)
        # Y = 0.299
        # Cb = -0.168736 + 0.5 = 0.331264
        # Cr = 0.5 + 0.5 = 1.0
        assert torch.isclose(ycbcr[0], torch.tensor(0.299), atol=1e-5)
        assert torch.isclose(ycbcr[1], torch.tensor(0.331264), atol=1e-5)
        assert torch.isclose(ycbcr[2], torch.tensor(1.0), atol=1e-5)

    def test_green(self):
        """Pure green: (0, 1, 0) -> (0.587, 0.168736, 0.081312)."""
        rgb = torch.tensor([0.0, 1.0, 0.0])
        ycbcr = srgb_to_ycbcr(rgb)
        # Y = 0.587
        # Cb = -0.331264 + 0.5 = 0.168736
        # Cr = -0.418688 + 0.5 = 0.081312
        assert torch.isclose(ycbcr[0], torch.tensor(0.587), atol=1e-5)
        assert torch.isclose(ycbcr[1], torch.tensor(0.168736), atol=1e-5)
        assert torch.isclose(ycbcr[2], torch.tensor(0.081312), atol=1e-5)

    def test_blue(self):
        """Pure blue: (0, 0, 1) -> (0.114, 1.0, 0.418688)."""
        rgb = torch.tensor([0.0, 0.0, 1.0])
        ycbcr = srgb_to_ycbcr(rgb)
        # Y = 0.114
        # Cb = 0.5 + 0.5 = 1.0
        # Cr = -0.081312 + 0.5 = 0.418688
        assert torch.isclose(ycbcr[0], torch.tensor(0.114), atol=1e-5)
        assert torch.isclose(ycbcr[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(ycbcr[2], torch.tensor(0.418688), atol=1e-5)

    def test_mid_gray(self):
        """Mid gray (0.5, 0.5, 0.5)."""
        rgb = torch.tensor([0.5, 0.5, 0.5])
        ycbcr = srgb_to_ycbcr(rgb)
        # Y = 0.5, Cb = 0.5, Cr = 0.5
        assert torch.isclose(ycbcr[0], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(ycbcr[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(ycbcr[2], torch.tensor(0.5), atol=1e-5)


class TestSrgbToYcbcrGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        rgb = torch.rand(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(srgb_to_ycbcr, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        rgb = torch.rand(4, 4, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(srgb_to_ycbcr, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        rgb = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)
        ycbcr = srgb_to_ycbcr(rgb)
        loss = ycbcr.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.any(torch.isnan(rgb.grad))


class TestSrgbToYcbcrDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        rgb = torch.rand(10, 3, dtype=torch.float32)
        ycbcr = srgb_to_ycbcr(rgb)
        assert ycbcr.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        rgb = torch.rand(10, 3, dtype=torch.float64)
        ycbcr = srgb_to_ycbcr(rgb)
        assert ycbcr.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        rgb = torch.rand(10, 3, dtype=torch.bfloat16)
        ycbcr = srgb_to_ycbcr(rgb)
        assert ycbcr.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        rgb = torch.rand(10, 3, dtype=torch.float16)
        ycbcr = srgb_to_ycbcr(rgb)
        assert ycbcr.dtype == torch.float16


class TestSrgbToYcbcrEdgeCases:
    """Tests for edge cases."""

    def test_values_outside_range(self):
        """Test with values outside [0, 1]."""
        rgb = torch.tensor([1.5, -0.5, 0.5])
        ycbcr = srgb_to_ycbcr(rgb)
        assert not torch.any(torch.isnan(ycbcr))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        rgb = torch.empty(0, 3)
        ycbcr = srgb_to_ycbcr(rgb)
        assert ycbcr.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        rgb = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not rgb.is_contiguous()
        ycbcr = srgb_to_ycbcr(rgb)
        assert ycbcr.shape == rgb.shape

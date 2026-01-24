"""Tests for srgb_to_yuv color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_yuv


class TestSrgbToYuvShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        yuv = srgb_to_yuv(rgb)
        assert yuv.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        rgb = torch.rand(10, 3)
        yuv = srgb_to_yuv(rgb)
        assert yuv.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        rgb = torch.rand(64, 64, 3)
        yuv = srgb_to_yuv(rgb)
        assert yuv.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        rgb = torch.rand(10, 32, 32, 3)
        yuv = srgb_to_yuv(rgb)
        assert yuv.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            srgb_to_yuv(torch.randn(10, 4))


class TestSrgbToYuvKnownValues:
    """Tests for known color conversions (BT.601 reference values)."""

    def test_white(self):
        """White: (1, 1, 1) -> (1, 0, 0)."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        yuv = srgb_to_yuv(rgb)
        # Y = 0.299 + 0.587 + 0.114 = 1.0
        # U = -0.14713 - 0.28886 + 0.436 = ~0
        # V = 0.615 - 0.51499 - 0.10001 = ~0
        assert torch.isclose(yuv[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(yuv[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(yuv[2], torch.tensor(0.0), atol=1e-4)

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0, 0)."""
        rgb = torch.tensor([0.0, 0.0, 0.0])
        yuv = srgb_to_yuv(rgb)
        assert torch.isclose(yuv[0], torch.tensor(0.0), atol=1e-7)
        assert torch.isclose(yuv[1], torch.tensor(0.0), atol=1e-7)
        assert torch.isclose(yuv[2], torch.tensor(0.0), atol=1e-7)

    def test_red(self):
        """Pure red: (1, 0, 0) -> (0.299, -0.1471376975, 0.615)."""
        rgb = torch.tensor([1.0, 0.0, 0.0])
        yuv = srgb_to_yuv(rgb)
        # Y = 0.299
        # U = -0.1471376975
        # V = 0.615
        assert torch.isclose(yuv[0], torch.tensor(0.299), atol=1e-5)
        assert torch.isclose(yuv[1], torch.tensor(-0.1471376975), atol=1e-5)
        assert torch.isclose(yuv[2], torch.tensor(0.615), atol=1e-5)

    def test_green(self):
        """Pure green: (0, 1, 0) -> (0.587, -0.2888623025, -0.5149857347)."""
        rgb = torch.tensor([0.0, 1.0, 0.0])
        yuv = srgb_to_yuv(rgb)
        # Y = 0.587
        # U = -0.2888623025
        # V = -0.5149857347
        assert torch.isclose(yuv[0], torch.tensor(0.587), atol=1e-5)
        assert torch.isclose(yuv[1], torch.tensor(-0.2888623025), atol=1e-5)
        assert torch.isclose(yuv[2], torch.tensor(-0.5149857347), atol=1e-5)

    def test_blue(self):
        """Pure blue: (0, 0, 1) -> (0.114, 0.436, -0.1000142653)."""
        rgb = torch.tensor([0.0, 0.0, 1.0])
        yuv = srgb_to_yuv(rgb)
        # Y = 0.114
        # U = 0.436
        # V = -0.1000142653
        assert torch.isclose(yuv[0], torch.tensor(0.114), atol=1e-5)
        assert torch.isclose(yuv[1], torch.tensor(0.436), atol=1e-5)
        assert torch.isclose(yuv[2], torch.tensor(-0.1000142653), atol=1e-5)

    def test_mid_gray(self):
        """Mid gray (0.5, 0.5, 0.5)."""
        rgb = torch.tensor([0.5, 0.5, 0.5])
        yuv = srgb_to_yuv(rgb)
        # Y = 0.5, U = 0, V = 0
        assert torch.isclose(yuv[0], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(yuv[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(yuv[2], torch.tensor(0.0), atol=1e-4)


class TestSrgbToYuvGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        rgb = torch.rand(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(srgb_to_yuv, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        rgb = torch.rand(4, 4, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(srgb_to_yuv, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        rgb = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)
        yuv = srgb_to_yuv(rgb)
        loss = yuv.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.any(torch.isnan(rgb.grad))


class TestSrgbToYuvDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        rgb = torch.rand(10, 3, dtype=torch.float32)
        yuv = srgb_to_yuv(rgb)
        assert yuv.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        rgb = torch.rand(10, 3, dtype=torch.float64)
        yuv = srgb_to_yuv(rgb)
        assert yuv.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        rgb = torch.rand(10, 3, dtype=torch.bfloat16)
        yuv = srgb_to_yuv(rgb)
        assert yuv.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        rgb = torch.rand(10, 3, dtype=torch.float16)
        yuv = srgb_to_yuv(rgb)
        assert yuv.dtype == torch.float16


class TestSrgbToYuvEdgeCases:
    """Tests for edge cases."""

    def test_values_outside_range(self):
        """Test with values outside [0, 1]."""
        rgb = torch.tensor([1.5, -0.5, 0.5])
        yuv = srgb_to_yuv(rgb)
        assert not torch.any(torch.isnan(yuv))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        rgb = torch.empty(0, 3)
        yuv = srgb_to_yuv(rgb)
        assert yuv.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        rgb = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not rgb.is_contiguous()
        yuv = srgb_to_yuv(rgb)
        assert yuv.shape == rgb.shape

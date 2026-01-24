"""Tests for srgb_to_ypbpr color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_ypbpr


class TestSrgbToYpbprShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        ypbpr = srgb_to_ypbpr(rgb)
        assert ypbpr.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        rgb = torch.rand(10, 3)
        ypbpr = srgb_to_ypbpr(rgb)
        assert ypbpr.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        rgb = torch.rand(64, 64, 3)
        ypbpr = srgb_to_ypbpr(rgb)
        assert ypbpr.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        rgb = torch.rand(10, 32, 32, 3)
        ypbpr = srgb_to_ypbpr(rgb)
        assert ypbpr.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            srgb_to_ypbpr(torch.randn(10, 4))


class TestSrgbToYpbprKnownValues:
    """Tests for known color conversions (BT.601 on linear RGB)."""

    def test_white(self):
        """White sRGB (1, 1, 1) linearizes to (1, 1, 1) -> Y=1, Pb=0, Pr=0."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        ypbpr = srgb_to_ypbpr(rgb)
        # Linear: (1, 1, 1)
        # Y = 0.299 + 0.587 + 0.114 = 1.0
        # Pb = -0.168736 - 0.331264 + 0.5 = 0.0
        # Pr = 0.5 - 0.418688 - 0.081312 = 0.0
        assert torch.isclose(ypbpr[0], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(ypbpr[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(ypbpr[2], torch.tensor(0.0), atol=1e-5)

    def test_black(self):
        """Black sRGB (0, 0, 0) -> Y=0, Pb=0, Pr=0."""
        rgb = torch.tensor([0.0, 0.0, 0.0])
        ypbpr = srgb_to_ypbpr(rgb)
        assert torch.isclose(ypbpr[0], torch.tensor(0.0), atol=1e-7)
        assert torch.isclose(ypbpr[1], torch.tensor(0.0), atol=1e-7)
        assert torch.isclose(ypbpr[2], torch.tensor(0.0), atol=1e-7)

    def test_red(self):
        """Pure red sRGB (1, 0, 0) linearizes to (1, 0, 0)."""
        rgb = torch.tensor([1.0, 0.0, 0.0])
        ypbpr = srgb_to_ypbpr(rgb)
        # Linear: (1, 0, 0)
        # Y = 0.299
        # Pb = -0.168736
        # Pr = 0.5
        assert torch.isclose(ypbpr[0], torch.tensor(0.299), atol=1e-5)
        assert torch.isclose(ypbpr[1], torch.tensor(-0.168736), atol=1e-5)
        assert torch.isclose(ypbpr[2], torch.tensor(0.5), atol=1e-5)

    def test_green(self):
        """Pure green sRGB (0, 1, 0) linearizes to (0, 1, 0)."""
        rgb = torch.tensor([0.0, 1.0, 0.0])
        ypbpr = srgb_to_ypbpr(rgb)
        # Linear: (0, 1, 0)
        # Y = 0.587
        # Pb = -0.331264
        # Pr = -0.418688
        assert torch.isclose(ypbpr[0], torch.tensor(0.587), atol=1e-5)
        assert torch.isclose(ypbpr[1], torch.tensor(-0.331264), atol=1e-5)
        assert torch.isclose(ypbpr[2], torch.tensor(-0.418688), atol=1e-5)

    def test_blue(self):
        """Pure blue sRGB (0, 0, 1) linearizes to (0, 0, 1)."""
        rgb = torch.tensor([0.0, 0.0, 1.0])
        ypbpr = srgb_to_ypbpr(rgb)
        # Linear: (0, 0, 1)
        # Y = 0.114
        # Pb = 0.5
        # Pr = -0.081312
        assert torch.isclose(ypbpr[0], torch.tensor(0.114), atol=1e-5)
        assert torch.isclose(ypbpr[1], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(ypbpr[2], torch.tensor(-0.081312), atol=1e-5)


class TestSrgbToYpbprGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        # Use values in the linear region to avoid gradient discontinuity
        rgb = (
            torch.rand(5, 3, dtype=torch.float64, requires_grad=True) * 0.8
            + 0.1
        )
        assert gradcheck(srgb_to_ypbpr, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        rgb = (
            torch.rand(4, 4, 3, dtype=torch.float64, requires_grad=True) * 0.8
            + 0.1
        )
        assert gradcheck(srgb_to_ypbpr, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        rgb = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)
        ypbpr = srgb_to_ypbpr(rgb)
        loss = ypbpr.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.any(torch.isnan(rgb.grad))


class TestSrgbToYpbprDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        rgb = torch.rand(10, 3, dtype=torch.float32)
        ypbpr = srgb_to_ypbpr(rgb)
        assert ypbpr.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        rgb = torch.rand(10, 3, dtype=torch.float64)
        ypbpr = srgb_to_ypbpr(rgb)
        assert ypbpr.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        rgb = torch.rand(10, 3, dtype=torch.bfloat16)
        ypbpr = srgb_to_ypbpr(rgb)
        assert ypbpr.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        rgb = torch.rand(10, 3, dtype=torch.float16)
        ypbpr = srgb_to_ypbpr(rgb)
        assert ypbpr.dtype == torch.float16


class TestSrgbToYpbprEdgeCases:
    """Tests for edge cases."""

    def test_values_outside_range(self):
        """Test with values outside [0, 1]."""
        rgb = torch.tensor([1.5, -0.5, 0.5])
        ypbpr = srgb_to_ypbpr(rgb)
        assert not torch.any(torch.isnan(ypbpr))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        rgb = torch.empty(0, 3)
        ypbpr = srgb_to_ypbpr(rgb)
        assert ypbpr.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        rgb = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not rgb.is_contiguous()
        ypbpr = srgb_to_ypbpr(rgb)
        assert ypbpr.shape == rgb.shape

    def test_pb_pr_range(self):
        """Verify Pb/Pr are in expected range [-0.5, 0.5] for valid inputs."""
        rgb = torch.rand(100, 3)
        ypbpr = srgb_to_ypbpr(rgb)
        # Y should be in [0, 1], Pb/Pr should be in approximately [-0.5, 0.5]
        assert torch.all(
            ypbpr[..., 0] >= -0.01
        )  # Y >= 0 (allow small numerical error)
        assert torch.all(ypbpr[..., 0] <= 1.01)  # Y <= 1
        assert torch.all(ypbpr[..., 1] >= -0.6)  # Pb >= -0.5
        assert torch.all(ypbpr[..., 1] <= 0.6)  # Pb <= 0.5
        assert torch.all(ypbpr[..., 2] >= -0.6)  # Pr >= -0.5
        assert torch.all(ypbpr[..., 2] <= 0.6)  # Pr <= 0.5

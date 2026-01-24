"""Tests for srgb_to_xyz color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_xyz


class TestSrgbToXyzShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        xyz = srgb_to_xyz(rgb)
        assert xyz.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        rgb = torch.randn(10, 3).abs()
        xyz = srgb_to_xyz(rgb)
        assert xyz.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        rgb = torch.randn(64, 64, 3).abs()
        xyz = srgb_to_xyz(rgb)
        assert xyz.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        rgb = torch.randn(10, 32, 32, 3).abs()
        xyz = srgb_to_xyz(rgb)
        assert xyz.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            srgb_to_xyz(torch.randn(10, 4))


class TestSrgbToXyzKnownValues:
    """Tests for known color conversions (reference: colour-science library)."""

    def test_white_d65(self):
        """D65 white point: (1, 1, 1) -> (0.9505, 1.0, 1.089)."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        xyz = srgb_to_xyz(rgb)
        # D65 white point in XYZ
        assert torch.isclose(xyz[0], torch.tensor(0.9505), atol=1e-3)
        assert torch.isclose(xyz[1], torch.tensor(1.0), atol=1e-3)
        assert torch.isclose(xyz[2], torch.tensor(1.0890), atol=1e-3)

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0, 0)."""
        rgb = torch.tensor([0.0, 0.0, 0.0])
        xyz = srgb_to_xyz(rgb)
        assert torch.allclose(xyz, torch.zeros(3), atol=1e-7)

    def test_red(self):
        """Pure red sRGB primary."""
        rgb = torch.tensor([1.0, 0.0, 0.0])
        xyz = srgb_to_xyz(rgb)
        # sRGB red primary in XYZ
        assert torch.isclose(xyz[0], torch.tensor(0.4124564), atol=1e-4)
        assert torch.isclose(xyz[1], torch.tensor(0.2126729), atol=1e-4)
        assert torch.isclose(xyz[2], torch.tensor(0.0193339), atol=1e-4)

    def test_green(self):
        """Pure green sRGB primary."""
        rgb = torch.tensor([0.0, 1.0, 0.0])
        xyz = srgb_to_xyz(rgb)
        assert torch.isclose(xyz[0], torch.tensor(0.3575761), atol=1e-4)
        assert torch.isclose(xyz[1], torch.tensor(0.7151522), atol=1e-4)
        assert torch.isclose(xyz[2], torch.tensor(0.1191920), atol=1e-4)

    def test_blue(self):
        """Pure blue sRGB primary."""
        rgb = torch.tensor([0.0, 0.0, 1.0])
        xyz = srgb_to_xyz(rgb)
        assert torch.isclose(xyz[0], torch.tensor(0.1804375), atol=1e-4)
        assert torch.isclose(xyz[1], torch.tensor(0.0721750), atol=1e-4)
        assert torch.isclose(xyz[2], torch.tensor(0.9503041), atol=1e-4)

    def test_mid_gray(self):
        """Mid gray (0.5, 0.5, 0.5)."""
        rgb = torch.tensor([0.5, 0.5, 0.5])
        xyz = srgb_to_xyz(rgb)
        # Y should be ~0.214 (linearized 0.5 is ~0.214)
        assert xyz[1] > 0.2 and xyz[1] < 0.25


class TestSrgbToXyzGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        rgb = (
            torch.rand(5, 3, dtype=torch.float64, requires_grad=True) * 0.9
            + 0.05
        )
        assert gradcheck(srgb_to_xyz, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        rgb = (
            torch.rand(4, 4, 3, dtype=torch.float64, requires_grad=True) * 0.9
            + 0.05
        )
        assert gradcheck(srgb_to_xyz, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        rgb = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)
        xyz = srgb_to_xyz(rgb)
        loss = xyz.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.any(torch.isnan(rgb.grad))


class TestSrgbToXyzDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        rgb = torch.rand(10, 3, dtype=torch.float32)
        xyz = srgb_to_xyz(rgb)
        assert xyz.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        rgb = torch.rand(10, 3, dtype=torch.float64)
        xyz = srgb_to_xyz(rgb)
        assert xyz.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        rgb = torch.rand(10, 3, dtype=torch.bfloat16)
        xyz = srgb_to_xyz(rgb)
        assert xyz.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        rgb = torch.rand(10, 3, dtype=torch.float16)
        xyz = srgb_to_xyz(rgb)
        assert xyz.dtype == torch.float16


class TestSrgbToXyzEdgeCases:
    """Tests for edge cases."""

    def test_hdr_values(self):
        """Test with HDR values > 1."""
        rgb = torch.tensor([1.5, 1.2, 0.8])
        xyz = srgb_to_xyz(rgb)
        assert not torch.any(torch.isnan(xyz))
        assert xyz[1] > 1.0  # Y should exceed 1 for HDR

    def test_negative_values(self):
        """Test with negative values (out of gamut)."""
        rgb = torch.tensor([-0.1, 0.5, 0.5])
        xyz = srgb_to_xyz(rgb)
        assert not torch.any(torch.isnan(xyz))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        rgb = torch.empty(0, 3)
        xyz = srgb_to_xyz(rgb)
        assert xyz.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        rgb = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not rgb.is_contiguous()
        xyz = srgb_to_xyz(rgb)
        assert xyz.shape == rgb.shape

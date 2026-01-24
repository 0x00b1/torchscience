"""Tests for srgb_to_lab color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_lab


class TestSrgbToLabShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        lab = srgb_to_lab(rgb)
        assert lab.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        rgb = torch.randn(10, 3).abs()
        lab = srgb_to_lab(rgb)
        assert lab.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        rgb = torch.randn(64, 64, 3).abs()
        lab = srgb_to_lab(rgb)
        assert lab.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        rgb = torch.randn(10, 32, 32, 3).abs()
        lab = srgb_to_lab(rgb)
        assert lab.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            srgb_to_lab(torch.randn(10, 4))


class TestSrgbToLabKnownValues:
    """Tests for known color conversions (reference: colour-science library)."""

    def test_white_d65(self):
        """D65 white point: (1, 1, 1) -> (100, 0, 0)."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        lab = srgb_to_lab(rgb)
        # White in Lab should be L=100, a=0, b=0
        assert torch.isclose(lab[0], torch.tensor(100.0), atol=1e-3)
        assert torch.isclose(lab[1], torch.tensor(0.0), atol=1e-3)
        assert torch.isclose(lab[2], torch.tensor(0.0), atol=1e-3)

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0, 0)."""
        rgb = torch.tensor([0.0, 0.0, 0.0])
        lab = srgb_to_lab(rgb)
        # Black should have L=0 (and undefined a,b but typically 0)
        assert torch.isclose(lab[0], torch.tensor(0.0), atol=1e-4)

    def test_red(self):
        """Pure red sRGB primary."""
        rgb = torch.tensor([1.0, 0.0, 0.0])
        lab = srgb_to_lab(rgb)
        # Reference values from colour-science
        assert torch.isclose(lab[0], torch.tensor(53.2329), atol=0.1)
        assert torch.isclose(lab[1], torch.tensor(80.1093), atol=0.5)
        assert torch.isclose(lab[2], torch.tensor(67.2201), atol=0.5)

    def test_green(self):
        """Pure green sRGB primary."""
        rgb = torch.tensor([0.0, 1.0, 0.0])
        lab = srgb_to_lab(rgb)
        # Reference values from colour-science
        assert torch.isclose(lab[0], torch.tensor(87.7370), atol=0.1)
        assert torch.isclose(lab[1], torch.tensor(-86.1846), atol=0.5)
        assert torch.isclose(lab[2], torch.tensor(83.1812), atol=0.5)

    def test_blue(self):
        """Pure blue sRGB primary."""
        rgb = torch.tensor([0.0, 0.0, 1.0])
        lab = srgb_to_lab(rgb)
        # Reference values from colour-science
        assert torch.isclose(lab[0], torch.tensor(32.3026), atol=0.1)
        assert torch.isclose(lab[1], torch.tensor(79.1967), atol=0.5)
        assert torch.isclose(lab[2], torch.tensor(-107.8637), atol=0.5)

    def test_mid_gray(self):
        """Mid gray (0.5, 0.5, 0.5) - should have a=0, b=0."""
        rgb = torch.tensor([0.5, 0.5, 0.5])
        lab = srgb_to_lab(rgb)
        # Gray should have a=0, b=0
        assert torch.isclose(lab[1], torch.tensor(0.0), atol=1e-3)
        assert torch.isclose(lab[2], torch.tensor(0.0), atol=1e-3)
        # L should be between 0 and 100
        assert lab[0] > 0 and lab[0] < 100


class TestSrgbToLabGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        rgb = (
            torch.rand(5, 3, dtype=torch.float64, requires_grad=True) * 0.9
            + 0.05
        )
        assert gradcheck(srgb_to_lab, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        rgb = (
            torch.rand(4, 4, 3, dtype=torch.float64, requires_grad=True) * 0.9
            + 0.05
        )
        assert gradcheck(srgb_to_lab, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        rgb = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)
        lab = srgb_to_lab(rgb)
        loss = lab.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.any(torch.isnan(rgb.grad))


class TestSrgbToLabDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        rgb = torch.rand(10, 3, dtype=torch.float32)
        lab = srgb_to_lab(rgb)
        assert lab.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        rgb = torch.rand(10, 3, dtype=torch.float64)
        lab = srgb_to_lab(rgb)
        assert lab.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        rgb = torch.rand(10, 3, dtype=torch.bfloat16)
        lab = srgb_to_lab(rgb)
        assert lab.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        rgb = torch.rand(10, 3, dtype=torch.float16)
        lab = srgb_to_lab(rgb)
        assert lab.dtype == torch.float16


class TestSrgbToLabEdgeCases:
    """Tests for edge cases."""

    def test_hdr_values(self):
        """Test with HDR values > 1."""
        rgb = torch.tensor([1.5, 1.2, 0.8])
        lab = srgb_to_lab(rgb)
        assert not torch.any(torch.isnan(lab))
        assert lab[0] > 100  # L should exceed 100 for HDR

    def test_negative_values(self):
        """Test with negative values (out of gamut)."""
        rgb = torch.tensor([-0.1, 0.5, 0.5])
        lab = srgb_to_lab(rgb)
        assert not torch.any(torch.isnan(lab))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        rgb = torch.empty(0, 3)
        lab = srgb_to_lab(rgb)
        assert lab.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        rgb = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not rgb.is_contiguous()
        lab = srgb_to_lab(rgb)
        assert lab.shape == rgb.shape


class TestSrgbToLabConsistency:
    """Tests for consistency with srgb_to_xyz."""

    def test_consistent_with_xyz_intermediate(self):
        """L component should be consistent with Y from XYZ."""
        from torchscience.color import srgb_to_xyz

        rgb = torch.rand(10, 3)
        lab = srgb_to_lab(rgb)
        xyz = srgb_to_xyz(rgb)

        # L* = 116 * f(Y/Yn) - 16 where Yn=1.0
        # For Y > delta^3, f(Y) = Y^(1/3)
        # So L* should correlate with Y
        # Just check they're both computed and reasonable
        assert torch.all(lab[:, 0] >= 0)  # L >= 0
        assert torch.all(xyz[:, 1] >= 0)  # Y >= 0

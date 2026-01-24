"""Tests for srgb_to_luv color conversion."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.color import srgb_to_luv


class TestSrgbToLuvShape:
    """Tests for shape handling."""

    def test_single_pixel(self):
        """Single pixel (3,) input."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        luv = srgb_to_luv(rgb)
        assert luv.shape == (3,)

    def test_batch_1d(self):
        """Batch of pixels (B, 3)."""
        rgb = torch.randn(10, 3).abs()
        luv = srgb_to_luv(rgb)
        assert luv.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, 3)."""
        rgb = torch.randn(64, 64, 3).abs()
        luv = srgb_to_luv(rgb)
        assert luv.shape == (64, 64, 3)

    def test_video_shape(self):
        """Video shape (T, H, W, 3)."""
        rgb = torch.randn(10, 32, 32, 3).abs()
        luv = srgb_to_luv(rgb)
        assert luv.shape == (10, 32, 32, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 3."""
        with pytest.raises(ValueError, match="last dimension 3"):
            srgb_to_luv(torch.randn(10, 4))


class TestSrgbToLuvKnownValues:
    """Tests for known color conversions (reference: colour-science library)."""

    def test_white_d65(self):
        """D65 white point: (1, 1, 1) -> (100, 0, 0)."""
        rgb = torch.tensor([1.0, 1.0, 1.0])
        luv = srgb_to_luv(rgb)
        # White in LUV should be L=100, u=0, v=0
        assert torch.isclose(luv[0], torch.tensor(100.0), atol=1e-3)
        assert torch.isclose(luv[1], torch.tensor(0.0), atol=1e-3)
        assert torch.isclose(luv[2], torch.tensor(0.0), atol=1e-3)

    def test_black(self):
        """Black: (0, 0, 0) -> (0, 0, 0)."""
        rgb = torch.tensor([0.0, 0.0, 0.0])
        luv = srgb_to_luv(rgb)
        # Black should have L=0 (and u,v are 0 by convention)
        assert torch.isclose(luv[0], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(luv[1], torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(luv[2], torch.tensor(0.0), atol=1e-4)

    def test_red(self):
        """Pure red sRGB primary."""
        rgb = torch.tensor([1.0, 0.0, 0.0])
        luv = srgb_to_luv(rgb)
        # Reference values from colour-science
        # L* should be same as Lab (53.2329)
        assert torch.isclose(luv[0], torch.tensor(53.2329), atol=0.1)
        # u* and v* for pure red
        assert torch.isclose(luv[1], torch.tensor(175.0150), atol=1.0)
        assert torch.isclose(luv[2], torch.tensor(37.7564), atol=1.0)

    def test_green(self):
        """Pure green sRGB primary."""
        rgb = torch.tensor([0.0, 1.0, 0.0])
        luv = srgb_to_luv(rgb)
        # Reference values from colour-science
        assert torch.isclose(luv[0], torch.tensor(87.7370), atol=0.1)
        assert torch.isclose(luv[1], torch.tensor(-83.0776), atol=1.0)
        assert torch.isclose(luv[2], torch.tensor(107.3985), atol=1.0)

    def test_blue(self):
        """Pure blue sRGB primary."""
        rgb = torch.tensor([0.0, 0.0, 1.0])
        luv = srgb_to_luv(rgb)
        # Reference values from colour-science
        assert torch.isclose(luv[0], torch.tensor(32.3026), atol=0.1)
        assert torch.isclose(luv[1], torch.tensor(-9.4054), atol=1.0)
        assert torch.isclose(luv[2], torch.tensor(-130.3423), atol=1.0)

    def test_mid_gray(self):
        """Mid gray (0.5, 0.5, 0.5) - should have u=0, v=0."""
        rgb = torch.tensor([0.5, 0.5, 0.5])
        luv = srgb_to_luv(rgb)
        # Gray should have u=0, v=0
        assert torch.isclose(luv[1], torch.tensor(0.0), atol=1e-3)
        assert torch.isclose(luv[2], torch.tensor(0.0), atol=1e-3)
        # L should be between 0 and 100
        assert luv[0] > 0 and luv[0] < 100


class TestSrgbToLuvGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Verify gradients with torch.autograd.gradcheck."""
        rgb = (
            torch.rand(5, 3, dtype=torch.float64, requires_grad=True) * 0.9
            + 0.05
        )
        assert gradcheck(srgb_to_luv, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check on batch."""
        rgb = (
            torch.rand(4, 4, 3, dtype=torch.float64, requires_grad=True) * 0.9
            + 0.05
        )
        assert gradcheck(srgb_to_luv, (rgb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradient flows through the operation."""
        rgb = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)
        luv = srgb_to_luv(rgb)
        loss = luv.sum()
        loss.backward()
        assert rgb.grad is not None
        assert not torch.any(torch.isnan(rgb.grad))


class TestSrgbToLuvDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Test with float32."""
        rgb = torch.rand(10, 3, dtype=torch.float32)
        luv = srgb_to_luv(rgb)
        assert luv.dtype == torch.float32

    def test_float64(self):
        """Test with float64."""
        rgb = torch.rand(10, 3, dtype=torch.float64)
        luv = srgb_to_luv(rgb)
        assert luv.dtype == torch.float64

    def test_bfloat16(self):
        """Test with bfloat16."""
        rgb = torch.rand(10, 3, dtype=torch.bfloat16)
        luv = srgb_to_luv(rgb)
        assert luv.dtype == torch.bfloat16

    def test_float16(self):
        """Test with float16."""
        rgb = torch.rand(10, 3, dtype=torch.float16)
        luv = srgb_to_luv(rgb)
        assert luv.dtype == torch.float16


class TestSrgbToLuvEdgeCases:
    """Tests for edge cases."""

    def test_hdr_values(self):
        """Test with HDR values > 1."""
        rgb = torch.tensor([1.5, 1.2, 0.8])
        luv = srgb_to_luv(rgb)
        assert not torch.any(torch.isnan(luv))
        assert luv[0] > 100  # L should exceed 100 for HDR

    def test_negative_values(self):
        """Test with negative values (out of gamut)."""
        rgb = torch.tensor([-0.1, 0.5, 0.5])
        luv = srgb_to_luv(rgb)
        assert not torch.any(torch.isnan(luv))

    def test_empty_tensor(self):
        """Test with empty tensor."""
        rgb = torch.empty(0, 3)
        luv = srgb_to_luv(rgb)
        assert luv.shape == (0, 3)

    def test_contiguous_memory(self):
        """Test with non-contiguous input."""
        rgb = torch.rand(10, 10, 3)[:, ::2, :]  # Non-contiguous
        assert not rgb.is_contiguous()
        luv = srgb_to_luv(rgb)
        assert luv.shape == rgb.shape


class TestSrgbToLuvConsistency:
    """Tests for consistency with related conversions."""

    def test_l_consistent_with_lab(self):
        """L* component should be identical to Lab L*."""
        from torchscience.color import srgb_to_lab

        rgb = torch.rand(10, 3)
        luv = srgb_to_luv(rgb)
        lab = srgb_to_lab(rgb)

        # L* should be the same in both Lab and Luv
        assert torch.allclose(luv[:, 0], lab[:, 0], atol=1e-4)

    def test_gray_has_zero_chroma(self):
        """Gray colors should have u*=0, v*=0."""
        grays = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
                [0.75, 0.75, 0.75],
                [1.0, 1.0, 1.0],
            ]
        )
        luv = srgb_to_luv(grays)

        # u* and v* should be 0 for all grays
        assert torch.allclose(luv[:, 1], torch.zeros(5), atol=1e-4)
        assert torch.allclose(luv[:, 2], torch.zeros(5), atol=1e-4)

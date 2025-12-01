"""Tests for srgb_to_srgb_linear color conversion."""

import torch
from torch.autograd import gradcheck

from torchscience.graphics.color import srgb_to_srgb_linear


class TestSrgbToSrgbLinearShape:
    """Tests for shape handling."""

    def test_single_value(self):
        """Single value (scalar-like) tensor."""
        srgb = torch.tensor([0.5])
        linear = srgb_to_srgb_linear(srgb)
        assert linear.shape == (1,)

    def test_rgb_vector(self):
        """RGB vector (3,) input."""
        srgb = torch.tensor([0.2, 0.5, 0.8])
        linear = srgb_to_srgb_linear(srgb)
        assert linear.shape == (3,)

    def test_batch_1d(self):
        """Batch of values (B,) or (B, C)."""
        srgb = torch.rand(10, 3)
        linear = srgb_to_srgb_linear(srgb)
        assert linear.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, C)."""
        srgb = torch.rand(64, 64, 3)
        linear = srgb_to_srgb_linear(srgb)
        assert linear.shape == (64, 64, 3)

    def test_batch_image_shape(self):
        """Batch of images (B, H, W, C)."""
        srgb = torch.rand(4, 32, 32, 3)
        linear = srgb_to_srgb_linear(srgb)
        assert linear.shape == (4, 32, 32, 3)


class TestSrgbToSrgbLinearKnownValues:
    """Tests for known conversion values based on IEC 61966-2-1."""

    def test_black(self):
        """sRGB 0.0 -> linear 0.0."""
        srgb = torch.tensor([0.0])
        linear = srgb_to_srgb_linear(srgb)
        assert torch.isclose(linear[0], torch.tensor(0.0), atol=1e-6)

    def test_white(self):
        """sRGB 1.0 -> linear 1.0."""
        srgb = torch.tensor([1.0])
        linear = srgb_to_srgb_linear(srgb)
        assert torch.isclose(linear[0], torch.tensor(1.0), atol=1e-6)

    def test_low_value_linear_region(self):
        """Values below threshold use linear formula: srgb / 12.92."""
        # 0.04045 is the threshold
        srgb = torch.tensor([0.01, 0.02, 0.04])
        linear = srgb_to_srgb_linear(srgb)
        expected = srgb / 12.92
        assert torch.allclose(linear, expected, atol=1e-6)

    def test_threshold_value(self):
        """Test value at threshold 0.04045."""
        srgb = torch.tensor([0.04045])
        linear = srgb_to_srgb_linear(srgb)
        # At threshold, linear formula: 0.04045 / 12.92 = 0.003130804954
        expected = torch.tensor([0.04045 / 12.92])
        assert torch.isclose(linear[0], expected[0], atol=1e-6)

    def test_above_threshold(self):
        """Values above threshold use gamma formula."""
        # For srgb = 0.5: ((0.5 + 0.055) / 1.055)^2.4 = 0.214041...
        srgb = torch.tensor([0.5])
        linear = srgb_to_srgb_linear(srgb)
        expected = ((0.5 + 0.055) / 1.055) ** 2.4
        assert torch.isclose(linear[0], torch.tensor(expected), atol=1e-5)

    def test_mid_gray(self):
        """Test mid gray value (sRGB 0.5)."""
        srgb = torch.tensor([0.5])
        linear = srgb_to_srgb_linear(srgb)
        # ((0.5 + 0.055) / 1.055)^2.4 = 0.2140411...
        expected = torch.tensor([0.2140411])
        assert torch.isclose(linear[0], expected[0], atol=1e-5)

    def test_known_rgb_values(self):
        """Test known RGB triplet conversion."""
        # sRGB (0.5, 0.5, 0.5) should convert to approximately (0.214, 0.214, 0.214)
        srgb = torch.tensor([0.5, 0.5, 0.5])
        linear = srgb_to_srgb_linear(srgb)
        expected_value = ((0.5 + 0.055) / 1.055) ** 2.4
        expected = torch.tensor(
            [expected_value, expected_value, expected_value]
        )
        assert torch.allclose(linear, expected, atol=1e-5)

    def test_mixed_regions(self):
        """Test values spanning both linear and gamma regions."""
        srgb = torch.tensor([0.01, 0.04045, 0.1, 0.5, 1.0])
        linear = srgb_to_srgb_linear(srgb)

        # Values in linear region
        assert torch.isclose(linear[0], torch.tensor(0.01 / 12.92), atol=1e-6)
        assert torch.isclose(
            linear[1], torch.tensor(0.04045 / 12.92), atol=1e-6
        )

        # Values in gamma region
        assert torch.isclose(
            linear[2], torch.tensor(((0.1 + 0.055) / 1.055) ** 2.4), atol=1e-5
        )
        assert torch.isclose(
            linear[3], torch.tensor(((0.5 + 0.055) / 1.055) ** 2.4), atol=1e-5
        )
        assert torch.isclose(linear[4], torch.tensor(1.0), atol=1e-6)


class TestSrgbToSrgbLinearGradients:
    """Tests for gradient computation."""

    def test_gradcheck_above_threshold(self):
        """Gradient check for values above threshold (gamma region)."""
        srgb = torch.tensor(
            [0.5, 0.7, 0.9], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(srgb_to_srgb_linear, (srgb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_below_threshold(self):
        """Gradient check for values below threshold (linear region)."""
        srgb = torch.tensor(
            [0.01, 0.02, 0.03], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(srgb_to_srgb_linear, (srgb,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check for batch of values."""
        torch.manual_seed(42)
        srgb = torch.rand(5, 3, dtype=torch.float64) * 0.8 + 0.1
        srgb.requires_grad_(True)
        assert gradcheck(srgb_to_srgb_linear, (srgb,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradients flow back correctly."""
        srgb = torch.tensor([0.3, 0.5, 0.7], requires_grad=True)
        linear = srgb_to_srgb_linear(srgb)
        loss = linear.sum()
        loss.backward()
        assert srgb.grad is not None
        assert not torch.isnan(srgb.grad).any()

    def test_gradient_linear_region(self):
        """Gradient in linear region should be 1/12.92."""
        srgb = torch.tensor([0.02], requires_grad=True)
        linear = srgb_to_srgb_linear(srgb)
        linear.backward()
        expected_grad = 1.0 / 12.92
        assert torch.isclose(
            srgb.grad[0], torch.tensor(expected_grad), atol=1e-5
        )

    def test_gradient_gamma_region(self):
        """Gradient in gamma region: (2.4 / 1.055) * ((srgb + 0.055) / 1.055)^1.4."""
        srgb_val = 0.5
        srgb = torch.tensor([srgb_val], requires_grad=True)
        linear = srgb_to_srgb_linear(srgb)
        linear.backward()
        expected_grad = (2.4 / 1.055) * ((srgb_val + 0.055) / 1.055) ** 1.4
        assert torch.isclose(
            srgb.grad[0], torch.tensor(expected_grad), atol=1e-5
        )


class TestSrgbToSrgbLinearDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        srgb = torch.rand(10, dtype=torch.float32)
        linear = srgb_to_srgb_linear(srgb)
        assert linear.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        srgb = torch.rand(10, dtype=torch.float64)
        linear = srgb_to_srgb_linear(srgb)
        assert linear.dtype == torch.float64

    def test_bfloat16(self):
        """Works with bfloat16."""
        srgb = torch.rand(10, dtype=torch.float32).to(torch.bfloat16)
        linear = srgb_to_srgb_linear(srgb)
        assert linear.dtype == torch.bfloat16

    def test_float16(self):
        """Works with float16."""
        srgb = torch.rand(10, dtype=torch.float32).to(torch.float16)
        linear = srgb_to_srgb_linear(srgb)
        assert linear.dtype == torch.float16


class TestSrgbToSrgbLinearEdgeCases:
    """Tests for edge cases."""

    def test_empty_tensor(self):
        """Empty tensor should return empty tensor."""
        srgb = torch.tensor([])
        linear = srgb_to_srgb_linear(srgb)
        assert linear.shape == (0,)

    def test_contiguous_memory(self):
        """Non-contiguous input should still work."""
        srgb = torch.rand(10, 10).T  # Transposed, non-contiguous
        linear = srgb_to_srgb_linear(srgb)
        assert linear.shape == srgb.shape

    def test_values_at_boundary(self):
        """Test values very close to threshold from both sides."""
        # Just below threshold
        srgb_below = torch.tensor([0.04044])
        linear_below = srgb_to_srgb_linear(srgb_below)
        expected_below = 0.04044 / 12.92
        assert torch.isclose(
            linear_below[0], torch.tensor(expected_below), atol=1e-6
        )

        # Just above threshold
        srgb_above = torch.tensor([0.04046])
        linear_above = srgb_to_srgb_linear(srgb_above)
        expected_above = ((0.04046 + 0.055) / 1.055) ** 2.4
        assert torch.isclose(
            linear_above[0], torch.tensor(expected_above), atol=1e-6
        )

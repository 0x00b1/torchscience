"""Tests for srgb_linear_to_srgb color conversion."""

import torch
from torch.autograd import gradcheck

from torchscience.color import (
    srgb_linear_to_srgb,
    srgb_to_srgb_linear,
)


class TestSrgbLinearToSrgbShape:
    """Tests for shape handling."""

    def test_single_value(self):
        """Single value (scalar-like) tensor."""
        linear = torch.tensor([0.5])
        srgb = srgb_linear_to_srgb(linear)
        assert srgb.shape == (1,)

    def test_rgb_vector(self):
        """RGB vector (3,) input."""
        linear = torch.tensor([0.2, 0.5, 0.8])
        srgb = srgb_linear_to_srgb(linear)
        assert srgb.shape == (3,)

    def test_batch_1d(self):
        """Batch of values (B,) or (B, C)."""
        linear = torch.rand(10, 3)
        srgb = srgb_linear_to_srgb(linear)
        assert srgb.shape == (10, 3)

    def test_image_shape(self):
        """Image shape (H, W, C)."""
        linear = torch.rand(64, 64, 3)
        srgb = srgb_linear_to_srgb(linear)
        assert srgb.shape == (64, 64, 3)

    def test_batch_image_shape(self):
        """Batch of images (B, H, W, C)."""
        linear = torch.rand(4, 32, 32, 3)
        srgb = srgb_linear_to_srgb(linear)
        assert srgb.shape == (4, 32, 32, 3)


class TestSrgbLinearToSrgbKnownValues:
    """Tests for known conversion values based on IEC 61966-2-1 inverse."""

    def test_black(self):
        """linear 0.0 -> sRGB 0.0."""
        linear = torch.tensor([0.0])
        srgb = srgb_linear_to_srgb(linear)
        assert torch.isclose(srgb[0], torch.tensor(0.0), atol=1e-6)

    def test_white(self):
        """linear 1.0 -> sRGB 1.0."""
        linear = torch.tensor([1.0])
        srgb = srgb_linear_to_srgb(linear)
        assert torch.isclose(srgb[0], torch.tensor(1.0), atol=1e-6)

    def test_low_value_linear_region(self):
        """Values below threshold use linear formula: linear * 12.92."""
        # 0.0031308 is the threshold for the inverse conversion
        linear = torch.tensor([0.001, 0.002, 0.003])
        srgb = srgb_linear_to_srgb(linear)
        expected = linear * 12.92
        assert torch.allclose(srgb, expected, atol=1e-6)

    def test_threshold_value(self):
        """Test value at threshold 0.0031308."""
        linear = torch.tensor([0.0031308])
        srgb = srgb_linear_to_srgb(linear)
        # At threshold, linear formula: 0.0031308 * 12.92 = 0.04045...
        expected = torch.tensor([0.0031308 * 12.92])
        assert torch.isclose(srgb[0], expected[0], atol=1e-5)

    def test_above_threshold(self):
        """Values above threshold use gamma formula."""
        # For linear = 0.214: 1.055 * 0.214^(1/2.4) - 0.055 ~ 0.5
        linear = torch.tensor([0.214041])
        srgb = srgb_linear_to_srgb(linear)
        expected = 1.055 * (0.214041 ** (1 / 2.4)) - 0.055
        assert torch.isclose(srgb[0], torch.tensor(expected), atol=1e-4)

    def test_mid_gray(self):
        """Test mid gray value (linear ~0.214 -> sRGB ~0.5)."""
        linear = torch.tensor([0.2140411])
        srgb = srgb_linear_to_srgb(linear)
        expected = torch.tensor([0.5])
        assert torch.isclose(srgb[0], expected[0], atol=1e-4)

    def test_known_rgb_values(self):
        """Test known RGB triplet conversion."""
        # linear (0.214, 0.214, 0.214) should convert to approximately (0.5, 0.5, 0.5)
        linear_val = 0.2140411
        linear = torch.tensor([linear_val, linear_val, linear_val])
        srgb = srgb_linear_to_srgb(linear)
        expected = torch.tensor([0.5, 0.5, 0.5])
        assert torch.allclose(srgb, expected, atol=1e-4)

    def test_mixed_regions(self):
        """Test values spanning both linear and gamma regions."""
        linear = torch.tensor([0.001, 0.0031308, 0.01, 0.214041, 1.0])
        srgb = srgb_linear_to_srgb(linear)

        # Values in linear region
        assert torch.isclose(srgb[0], torch.tensor(0.001 * 12.92), atol=1e-5)
        assert torch.isclose(
            srgb[1], torch.tensor(0.0031308 * 12.92), atol=1e-5
        )

        # Values in gamma region
        assert torch.isclose(
            srgb[2],
            torch.tensor(1.055 * (0.01 ** (1 / 2.4)) - 0.055),
            atol=1e-4,
        )
        assert torch.isclose(
            srgb[3],
            torch.tensor(1.055 * (0.214041 ** (1 / 2.4)) - 0.055),
            atol=1e-4,
        )
        assert torch.isclose(srgb[4], torch.tensor(1.0), atol=1e-5)


class TestSrgbLinearToSrgbRoundTrip:
    """Tests for round-trip conversion (linear -> srgb -> linear = identity)."""

    def test_round_trip_above_threshold(self):
        """Round-trip for values in gamma region."""
        linear_original = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        srgb = srgb_linear_to_srgb(linear_original)
        linear_recovered = srgb_to_srgb_linear(srgb)
        assert torch.allclose(linear_original, linear_recovered, atol=1e-5)

    def test_round_trip_below_threshold(self):
        """Round-trip for values in linear region."""
        linear_original = torch.tensor([0.0001, 0.001, 0.002, 0.003])
        srgb = srgb_linear_to_srgb(linear_original)
        linear_recovered = srgb_to_srgb_linear(srgb)
        assert torch.allclose(linear_original, linear_recovered, atol=1e-6)

    def test_round_trip_mixed(self):
        """Round-trip for values spanning both regions."""
        linear_original = torch.tensor([0.001, 0.01, 0.1, 0.5, 1.0])
        srgb = srgb_linear_to_srgb(linear_original)
        linear_recovered = srgb_to_srgb_linear(srgb)
        assert torch.allclose(linear_original, linear_recovered, atol=1e-5)

    def test_round_trip_batch(self):
        """Round-trip for batch of random values."""
        torch.manual_seed(42)
        linear_original = torch.rand(100, 3)
        srgb = srgb_linear_to_srgb(linear_original)
        linear_recovered = srgb_to_srgb_linear(srgb)
        assert torch.allclose(linear_original, linear_recovered, atol=1e-5)

    def test_inverse_round_trip(self):
        """Inverse round-trip (srgb -> linear -> srgb = identity)."""
        torch.manual_seed(42)
        srgb_original = torch.rand(100, 3)
        linear = srgb_to_srgb_linear(srgb_original)
        srgb_recovered = srgb_linear_to_srgb(linear)
        assert torch.allclose(srgb_original, srgb_recovered, atol=1e-5)


class TestSrgbLinearToSrgbGradients:
    """Tests for gradient computation."""

    def test_gradcheck_above_threshold(self):
        """Gradient check for values above threshold (gamma region)."""
        linear = torch.tensor(
            [0.1, 0.3, 0.5], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(srgb_linear_to_srgb, (linear,), eps=1e-6, atol=1e-4)

    def test_gradcheck_below_threshold(self):
        """Gradient check for values below threshold (linear region)."""
        linear = torch.tensor(
            [0.001, 0.002, 0.003], dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(srgb_linear_to_srgb, (linear,), eps=1e-6, atol=1e-4)

    def test_gradcheck_batch(self):
        """Gradient check for batch of values."""
        torch.manual_seed(42)
        linear = torch.rand(5, 3, dtype=torch.float64) * 0.8 + 0.1
        linear.requires_grad_(True)
        assert gradcheck(srgb_linear_to_srgb, (linear,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Verify gradients flow back correctly."""
        linear = torch.tensor([0.1, 0.3, 0.5], requires_grad=True)
        srgb = srgb_linear_to_srgb(linear)
        loss = srgb.sum()
        loss.backward()
        assert linear.grad is not None
        assert not torch.isnan(linear.grad).any()

    def test_gradient_linear_region(self):
        """Gradient in linear region should be 12.92."""
        linear = torch.tensor([0.002], requires_grad=True)
        srgb = srgb_linear_to_srgb(linear)
        srgb.backward()
        expected_grad = 12.92
        assert torch.isclose(
            linear.grad[0], torch.tensor(expected_grad), atol=1e-4
        )

    def test_gradient_gamma_region(self):
        """Gradient in gamma region: (1.055 / 2.4) * linear^(1/2.4 - 1)."""
        linear_val = 0.3
        linear = torch.tensor([linear_val], requires_grad=True)
        srgb = srgb_linear_to_srgb(linear)
        srgb.backward()
        # d_srgb/d_linear = (1.055 / 2.4) * linear^(1/2.4 - 1)
        expected_grad = (1.055 / 2.4) * (linear_val ** (1 / 2.4 - 1))
        assert torch.isclose(
            linear.grad[0], torch.tensor(expected_grad), atol=1e-4
        )


class TestSrgbLinearToSrgbDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        linear = torch.rand(10, dtype=torch.float32)
        srgb = srgb_linear_to_srgb(linear)
        assert srgb.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        linear = torch.rand(10, dtype=torch.float64)
        srgb = srgb_linear_to_srgb(linear)
        assert srgb.dtype == torch.float64

    def test_bfloat16(self):
        """Works with bfloat16."""
        linear = torch.rand(10, dtype=torch.float32).to(torch.bfloat16)
        srgb = srgb_linear_to_srgb(linear)
        assert srgb.dtype == torch.bfloat16

    def test_float16(self):
        """Works with float16."""
        linear = torch.rand(10, dtype=torch.float32).to(torch.float16)
        srgb = srgb_linear_to_srgb(linear)
        assert srgb.dtype == torch.float16


class TestSrgbLinearToSrgbEdgeCases:
    """Tests for edge cases."""

    def test_empty_tensor(self):
        """Empty tensor should return empty tensor."""
        linear = torch.tensor([])
        srgb = srgb_linear_to_srgb(linear)
        assert srgb.shape == (0,)

    def test_contiguous_memory(self):
        """Non-contiguous input should still work."""
        linear = torch.rand(10, 10).T  # Transposed, non-contiguous
        srgb = srgb_linear_to_srgb(linear)
        assert srgb.shape == linear.shape

    def test_values_at_boundary(self):
        """Test values very close to threshold from both sides."""
        # Just below threshold
        linear_below = torch.tensor([0.0031307])
        srgb_below = srgb_linear_to_srgb(linear_below)
        expected_below = 0.0031307 * 12.92
        assert torch.isclose(
            srgb_below[0], torch.tensor(expected_below), atol=1e-6
        )

        # Just above threshold
        linear_above = torch.tensor([0.0031309])
        srgb_above = srgb_linear_to_srgb(linear_above)
        expected_above = 1.055 * (0.0031309 ** (1 / 2.4)) - 0.055
        assert torch.isclose(
            srgb_above[0], torch.tensor(expected_above), atol=1e-5
        )

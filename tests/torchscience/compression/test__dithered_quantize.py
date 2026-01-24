"""Tests for dithered quantization."""

import pytest
import torch

from torchscience.compression import dithered_quantize


class TestDitheredQuantizeBasic:
    """Basic functionality tests."""

    def test_output_types(self):
        """Returns quantized tensor, indices, and dither."""
        x = torch.linspace(0, 1, 100)
        q, idx, dither = dithered_quantize(x, levels=8)
        assert isinstance(q, torch.Tensor)
        assert isinstance(idx, torch.Tensor)
        assert idx.dtype == torch.long

    def test_output_shapes(self):
        """Output shapes match input."""
        x = torch.randn(3, 4, 5)
        q, idx, dither = dithered_quantize(x, levels=16)
        assert q.shape == x.shape
        assert idx.shape == x.shape
        if dither is not None:
            assert dither.shape == x.shape

    def test_indices_in_range(self):
        """Indices are within valid range."""
        x = torch.randn(100)
        levels = 16
        q, idx, _ = dithered_quantize(x, levels=levels)
        assert idx.min() >= 0
        assert idx.max() < levels


class TestDitheredQuantizeSubtractive:
    """Tests for subtractive dithering."""

    def test_subtractive_returns_dither(self):
        """Subtractive mode returns dither signal."""
        x = torch.randn(50)
        _, _, dither = dithered_quantize(x, dither_type="subtractive")
        assert dither is not None

    def test_subtractive_different_from_none(self):
        """Subtractive dithering produces different result than none."""
        torch.manual_seed(42)
        x = torch.linspace(0, 1, 100)

        torch.manual_seed(42)
        q_sub, _, _ = dithered_quantize(x, levels=8, dither_type="subtractive")

        torch.manual_seed(42)
        q_none, _, _ = dithered_quantize(x, levels=8, dither_type="none")

        # Results should differ
        assert not torch.allclose(q_sub, q_none)


class TestDitheredQuantizeAdditive:
    """Tests for additive dithering."""

    def test_additive_returns_dither(self):
        """Additive mode returns dither signal."""
        x = torch.randn(50)
        _, _, dither = dithered_quantize(x, dither_type="additive")
        assert dither is not None

    def test_additive_quantized_is_discrete(self):
        """Additive dithering produces discrete quantized values."""
        x = torch.linspace(0, 1, 100)
        q, _, _ = dithered_quantize(x, levels=4, dither_type="additive")
        # Should have at most 4 unique values (might have fewer at edges)
        unique = q.unique()
        assert len(unique) <= 4


class TestDitheredQuantizeNoDither:
    """Tests for no dithering mode."""

    def test_none_returns_no_dither(self):
        """None mode returns no dither signal."""
        x = torch.randn(50)
        _, _, dither = dithered_quantize(x, dither_type="none")
        assert dither is None

    def test_none_is_standard_quantization(self):
        """None mode equals standard uniform quantization."""
        x = torch.linspace(0, 1, 100)
        q, idx, _ = dithered_quantize(x, levels=4, dither_type="none")

        # Should have exactly 4 unique values
        unique = q.unique()
        assert len(unique) == 4


class TestDitheredQuantizeNoiseTypes:
    """Tests for different noise types."""

    def test_uniform_noise(self):
        """Uniform noise type works."""
        x = torch.randn(100)
        q, _, dither = dithered_quantize(x, noise_type="uniform")
        assert dither is not None

    def test_triangular_noise(self):
        """Triangular noise type works."""
        x = torch.randn(100)
        q, _, dither = dithered_quantize(x, noise_type="triangular")
        assert dither is not None

    def test_triangular_wider_distribution(self):
        """Triangular dither has wider distribution than uniform."""
        torch.manual_seed(42)
        x = torch.randn(10000)

        torch.manual_seed(42)
        _, _, dither_uniform = dithered_quantize(x, noise_type="uniform")

        torch.manual_seed(42)
        _, _, dither_tri = dithered_quantize(x, noise_type="triangular")

        # Triangular PDF has wider support (sum of two uniforms)
        # Its std should be larger
        assert dither_tri.std() > dither_uniform.std()


class TestDitheredQuantizeProperties:
    """Tests for dithering properties."""

    def test_subtractive_error_decorrelated(self):
        """Subtractive dithering decorrelates error from input."""
        torch.manual_seed(42)
        x = torch.linspace(0, 1, 1000)
        q, _, _ = dithered_quantize(x, levels=8, dither_type="subtractive")
        error = q - x

        # Error should be roughly uncorrelated with input
        # (not a perfect test, but checks the property)
        correlation = torch.corrcoef(torch.stack([x, error]))[0, 1]
        # Correlation should be small (not exactly 0 due to finite samples)
        assert correlation.abs() < 0.2

    def test_preserves_mean_approximately(self):
        """Quantization approximately preserves signal mean."""
        torch.manual_seed(42)
        x = torch.randn(1000)
        q, _, _ = dithered_quantize(x, levels=64, dither_type="subtractive")

        # Mean should be approximately preserved
        assert torch.isclose(x.mean(), q.mean(), atol=0.1)


class TestDitheredQuantizeEdgeCases:
    """Edge case tests."""

    def test_constant_input(self):
        """Handles constant input."""
        x = torch.ones(10) * 0.5
        q, idx, dither = dithered_quantize(x, levels=4)
        assert torch.allclose(q, x)
        assert dither is None  # No dithering for constant

    def test_single_element(self):
        """Works with single element."""
        x = torch.tensor([0.5])
        q, idx, _ = dithered_quantize(x, levels=4, dither_type="none")
        assert q.shape == x.shape

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            dithered_quantize([0.1, 0.5], levels=4)

    def test_invalid_levels_raises(self):
        """Raises error for invalid levels."""
        x = torch.randn(10)
        with pytest.raises(ValueError, match="levels"):
            dithered_quantize(x, levels=1)

    def test_invalid_noise_type_raises(self):
        """Raises error for invalid noise type."""
        x = torch.randn(10)
        with pytest.raises(ValueError, match="noise_type"):
            dithered_quantize(x, noise_type="invalid")


class TestDitheredQuantizeDevice:
    """Device compatibility tests."""

    def test_cpu(self):
        """Works on CPU."""
        x = torch.randn(10, device="cpu")
        q, idx, dither = dithered_quantize(x, levels=8)
        assert q.device.type == "cpu"
        assert idx.device.type == "cpu"
        if dither is not None:
            assert dither.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        x = torch.randn(10, device="cuda")
        q, idx, dither = dithered_quantize(x, levels=8)
        assert q.device.type == "cuda"
        assert idx.device.type == "cuda"
        if dither is not None:
            assert dither.device.type == "cuda"

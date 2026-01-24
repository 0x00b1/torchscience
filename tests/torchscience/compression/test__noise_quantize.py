"""Tests for noise-based quantization."""

import pytest
import torch

from torchscience.compression import noise_quantize, soft_round, ste_round


class TestNoiseQuantize:
    """Tests for noise_quantize function."""

    def test_output_shape(self):
        """Preserves input shape."""
        x = torch.randn(4, 32, 8, 8)
        y = noise_quantize(x)
        assert y.shape == x.shape

    def test_training_adds_noise(self):
        """Training mode adds noise."""
        x = torch.randn(10, 32)

        torch.manual_seed(42)
        y1 = noise_quantize(x, training=True)
        torch.manual_seed(43)
        y2 = noise_quantize(x, training=True)

        # Different seeds should give different outputs
        assert not torch.allclose(y1, y2)

    def test_eval_quantizes(self):
        """Eval mode rounds to integers."""
        x = torch.randn(10, 32) * 5  # Scale up for clear quantization
        y = noise_quantize(x, training=False)
        assert torch.allclose(y, y.round())

    def test_uniform_noise_range(self):
        """Uniform noise is in [-0.5, 0.5)."""
        x = torch.zeros(10000)  # Zero input
        y = noise_quantize(x, training=True, noise_type="uniform")
        noise = y - x

        # Should be approximately uniform on [-0.5, 0.5]
        assert noise.min() >= -0.5
        assert noise.max() < 0.5
        assert abs(noise.mean()) < 0.02  # Mean should be near 0

    def test_triangular_noise_range(self):
        """Triangular noise has expected range."""
        x = torch.zeros(10000)
        y = noise_quantize(x, training=True, noise_type="triangular")
        noise = y - x

        # Triangular scaled to [-0.5, 0.5]
        assert noise.min() >= -0.5
        assert noise.max() <= 0.5

    def test_triangular_vs_uniform_distribution(self):
        """Triangular noise has different distribution than uniform."""
        torch.manual_seed(42)
        x = torch.zeros(100000)

        y_uniform = noise_quantize(x, training=True, noise_type="uniform")
        y_triangular = noise_quantize(
            x, training=True, noise_type="triangular"
        )

        # Triangular has lower kurtosis (more peaked)
        uniform_kurtosis = (y_uniform**4).mean() / (y_uniform**2).mean() ** 2
        triangular_kurtosis = (y_triangular**4).mean() / (
            y_triangular**2
        ).mean() ** 2

        # Uniform kurtosis ≈ 1.8, triangular ≈ 2.4 (for standard [-0.5, 0.5])
        assert triangular_kurtosis > uniform_kurtosis

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            noise_quantize([1.0, 2.0])

    def test_invalid_noise_type_raises(self):
        """Raises error for invalid noise type."""
        x = torch.randn(10)
        with pytest.raises(ValueError, match="noise_type"):
            noise_quantize(x, noise_type="invalid")


class TestSteRound:
    """Tests for ste_round function."""

    def test_forward_rounds(self):
        """Forward pass rounds values."""
        x = torch.tensor([1.2, 2.7, -0.3, -1.8])
        y = ste_round(x)
        expected = torch.tensor([1.0, 3.0, 0.0, -2.0])
        assert torch.allclose(y, expected)

    def test_gradient_passes_through(self):
        """Gradients pass through unchanged."""
        x = torch.tensor([1.2, 2.7, -0.3], requires_grad=True)
        y = ste_round(x)
        y.backward(torch.ones_like(y))
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_gradient_custom_upstream(self):
        """Custom upstream gradients pass through."""
        x = torch.tensor([1.5, 2.5], requires_grad=True)
        y = ste_round(x)
        upstream = torch.tensor([2.0, 3.0])
        y.backward(upstream)
        assert torch.allclose(x.grad, upstream)

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            ste_round([1.0, 2.0])


class TestSoftRound:
    """Tests for soft_round function."""

    def test_output_shape(self):
        """Preserves input shape."""
        x = torch.randn(4, 8, 16)
        y = soft_round(x)
        assert y.shape == x.shape

    def test_low_temperature_approaches_round(self):
        """Low temperature approaches hard rounding."""
        x = torch.tensor([0.3, 0.7, 1.2, 1.8])
        y_soft = soft_round(x, temperature=0.01)
        y_hard = torch.round(x)
        assert torch.allclose(y_soft, y_hard, atol=0.01)

    def test_high_temperature_is_smooth(self):
        """High temperature gives smooth output within each period."""
        # Test within a single integer interval (0 to 1) to avoid floor discontinuities
        x = torch.linspace(0.01, 0.99, 100)
        y = soft_round(x, temperature=10.0)

        # With high temperature, output should be smooth within the interval
        diffs = torch.abs(y[1:] - y[:-1])
        assert diffs.max() < 0.02

    def test_at_half_integers(self):
        """At half integers, output is between floor and ceil."""
        x = torch.tensor([0.5, 1.5, 2.5])
        y = soft_round(x, temperature=1.0)

        # At x.5, sigmoid((0.5 - 0.5)/1) = sigmoid(0) = 0.5
        # So output should be floor(x) + 0.5
        expected = torch.tensor([0.5, 1.5, 2.5])
        assert torch.allclose(y, expected)

    def test_differentiable(self):
        """Has non-zero gradients."""
        x = torch.tensor([0.3, 0.7, 1.5], requires_grad=True)
        y = soft_round(x, temperature=1.0)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x))

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            soft_round([1.0, 2.0])

    def test_invalid_temperature_raises(self):
        """Raises error for non-positive temperature."""
        x = torch.randn(10)
        with pytest.raises(ValueError, match="temperature"):
            soft_round(x, temperature=0)
        with pytest.raises(ValueError, match="temperature"):
            soft_round(x, temperature=-1)


class TestNoiseQuantizeDevice:
    """Device compatibility tests."""

    def test_noise_quantize_cpu(self):
        """noise_quantize works on CPU."""
        x = torch.randn(10, 32, device="cpu")
        y = noise_quantize(x)
        assert y.device.type == "cpu"

    def test_ste_round_cpu(self):
        """ste_round works on CPU."""
        x = torch.randn(10, device="cpu")
        y = ste_round(x)
        assert y.device.type == "cpu"

    def test_soft_round_cpu(self):
        """soft_round works on CPU."""
        x = torch.randn(10, device="cpu")
        y = soft_round(x)
        assert y.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_noise_quantize_cuda(self):
        """noise_quantize works on CUDA."""
        x = torch.randn(10, 32, device="cuda")
        y = noise_quantize(x)
        assert y.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_ste_round_cuda(self):
        """ste_round works on CUDA."""
        x = torch.randn(10, device="cuda")
        y = ste_round(x)
        assert y.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_soft_round_cuda(self):
        """soft_round works on CUDA."""
        x = torch.randn(10, device="cuda")
        y = soft_round(x)
        assert y.device.type == "cuda"

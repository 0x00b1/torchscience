"""Tests for Kraskov entropy estimator (alias for Kozachenko-Leonenko)."""

import math

import pytest
import torch

from torchscience.information import (
    kozachenko_leonenko_entropy,
    kraskov_entropy,
)


class TestKraskovEntropyBasic:
    """Basic functionality tests."""

    def test_is_alias_for_kl(self):
        """kraskov_entropy is equivalent to kozachenko_leonenko_entropy with k=3."""
        torch.manual_seed(42)
        samples = torch.randn(500, 3)
        h_kraskov = kraskov_entropy(samples)
        h_kl = kozachenko_leonenko_entropy(samples, k=3)
        assert torch.equal(h_kraskov, h_kl)

    def test_default_k_is_3(self):
        """Default k value is 3."""
        torch.manual_seed(42)
        samples = torch.randn(500, 2)
        h_default = kraskov_entropy(samples)
        h_k3 = kraskov_entropy(samples, k=3)
        assert torch.equal(h_default, h_k3)

    def test_output_shape_2d(self):
        """Returns scalar for (n_samples, n_dims) input."""
        samples = torch.randn(1000, 2)
        result = kraskov_entropy(samples)
        assert result.shape == torch.Size([])

    def test_output_shape_3d_batch(self):
        """Returns 1D tensor for batched input."""
        samples = torch.randn(5, 1000, 3)
        result = kraskov_entropy(samples)
        assert result.shape == torch.Size([5])


class TestKraskovEntropyCorrectness:
    """Numerical correctness tests."""

    def test_gaussian_entropy(self):
        """Matches expected Gaussian entropy."""
        torch.manual_seed(42)
        sigma = 1.0
        d = 2
        n_samples = 5000
        samples = torch.randn(n_samples, d) * sigma
        result = kraskov_entropy(samples)
        expected = d / 2.0 * math.log(2 * math.pi * math.e * sigma**2)
        assert torch.isclose(result, torch.tensor(expected), rtol=0.15), (
            f"Expected {expected}, got {result}"
        )

    def test_custom_k_value(self):
        """Works with custom k values."""
        torch.manual_seed(42)
        samples = torch.randn(500, 2)
        h_k1 = kraskov_entropy(samples, k=1)
        h_k5 = kraskov_entropy(samples, k=5)
        # Both should be finite and similar
        assert torch.isfinite(h_k1)
        assert torch.isfinite(h_k5)


class TestKraskovEntropyBase:
    """Tests for logarithm base conversion."""

    def test_base_conversion(self):
        """Base parameter works correctly."""
        torch.manual_seed(42)
        samples = torch.randn(500, 2)
        h_nats = kraskov_entropy(samples)
        h_bits = kraskov_entropy(samples, base=2)
        expected_bits = h_nats / math.log(2)
        assert torch.isclose(h_bits, expected_bits, rtol=1e-5)


class TestKraskovEntropyEdgeCases:
    """Tests for edge cases."""

    def test_k_too_large_raises(self):
        """Raises error when k >= n_samples."""
        samples = torch.randn(10, 2)
        with pytest.raises(ValueError, match="k must be less than n_samples"):
            kraskov_entropy(samples, k=10)

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            kraskov_entropy([[1, 2], [3, 4]])


class TestKraskovEntropyGradients:
    """Tests for gradient computation."""

    def test_gradients_exist(self):
        """Gradients flow through the computation."""
        torch.manual_seed(42)
        samples = torch.randn(100, 2, requires_grad=True)
        result = kraskov_entropy(samples)
        result.backward()
        assert samples.grad is not None


class TestKraskovEntropyDeviceDtype:
    """Tests for device and dtype handling."""

    def test_float64(self):
        """Works with float64."""
        samples = torch.randn(500, 2, dtype=torch.float64)
        result = kraskov_entropy(samples)
        assert result.dtype == torch.float64

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        samples = torch.randn(500, 2, device="cuda")
        result = kraskov_entropy(samples)
        assert result.device.type == "cuda"

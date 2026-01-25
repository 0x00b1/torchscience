"""Tests for kernel density estimation-based entropy."""

import math

import pytest
import torch

from torchscience.information import kernel_density_entropy


class TestKernelDensityEntropyBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Returns scalar for (n_samples, n_dims) input."""
        samples = torch.randn(1000, 2)
        result = kernel_density_entropy(samples)
        assert result.shape == torch.Size([])

    def test_output_shape_3d_batch(self):
        """Returns 1D tensor for batched input."""
        samples = torch.randn(5, 1000, 3)
        result = kernel_density_entropy(samples)
        assert result.shape == torch.Size([5])

    def test_output_is_finite(self):
        """Result is finite for normal inputs."""
        torch.manual_seed(42)
        samples = torch.randn(500, 3)
        result = kernel_density_entropy(samples)
        assert torch.isfinite(result)


class TestKernelDensityEntropyCorrectness:
    """Numerical correctness tests."""

    def test_gaussian_1d_entropy(self):
        """1D Gaussian entropy: H = 0.5 * log(2 * pi * e * sigma^2)."""
        torch.manual_seed(42)
        sigma = 1.0
        n_samples = 5000
        samples = torch.randn(n_samples, 1) * sigma
        result = kernel_density_entropy(samples)
        expected = 0.5 * math.log(2 * math.pi * math.e * sigma**2)
        # KDE can have significant bias depending on bandwidth
        assert torch.isclose(result, torch.tensor(expected), rtol=0.25), (
            f"Expected {expected}, got {result}"
        )

    def test_gaussian_2d_entropy(self):
        """2D Gaussian entropy: H = d/2 * log(2 * pi * e * sigma^2)."""
        torch.manual_seed(42)
        sigma = 1.0
        d = 2
        n_samples = 5000
        samples = torch.randn(n_samples, d) * sigma
        result = kernel_density_entropy(samples)
        expected = d / 2.0 * math.log(2 * math.pi * math.e * sigma**2)
        assert torch.isclose(result, torch.tensor(expected), rtol=0.25), (
            f"Expected {expected}, got {result}"
        )

    def test_larger_sigma_higher_entropy(self):
        """Larger sigma gives higher entropy."""
        torch.manual_seed(42)
        n = 2000
        small_sigma = torch.randn(n, 1) * 0.5
        large_sigma = torch.randn(n, 1) * 2.0
        h_small = kernel_density_entropy(small_sigma)
        h_large = kernel_density_entropy(large_sigma)
        assert h_large > h_small, (
            f"Larger sigma should give higher entropy: {h_large} vs {h_small}"
        )


class TestKernelDensityEntropyBandwidth:
    """Tests for bandwidth parameter."""

    def test_scott_rule(self):
        """Scott's rule bandwidth works."""
        torch.manual_seed(42)
        samples = torch.randn(500, 2)
        result = kernel_density_entropy(samples, bandwidth="scott")
        assert torch.isfinite(result)

    def test_silverman_rule(self):
        """Silverman's rule bandwidth works."""
        torch.manual_seed(42)
        samples = torch.randn(500, 2)
        result = kernel_density_entropy(samples, bandwidth="silverman")
        assert torch.isfinite(result)

    def test_custom_bandwidth(self):
        """Custom float bandwidth works."""
        torch.manual_seed(42)
        samples = torch.randn(500, 2)
        result = kernel_density_entropy(samples, bandwidth=0.5)
        assert torch.isfinite(result)

    def test_bandwidth_effect(self):
        """Smaller bandwidth gives different result than larger."""
        torch.manual_seed(42)
        samples = torch.randn(500, 2)
        h_small = kernel_density_entropy(samples, bandwidth=0.1)
        h_large = kernel_density_entropy(samples, bandwidth=1.0)
        # Results should be different (larger bandwidth -> lower entropy typically)
        assert not torch.isclose(h_small, h_large, rtol=0.1)

    def test_invalid_bandwidth_method_raises(self):
        """Invalid bandwidth method raises error."""
        samples = torch.randn(100, 2)
        with pytest.raises(ValueError, match="bandwidth must be"):
            kernel_density_entropy(samples, bandwidth="invalid")

    def test_negative_bandwidth_raises(self):
        """Negative bandwidth raises error."""
        samples = torch.randn(100, 2)
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            kernel_density_entropy(samples, bandwidth=-0.5)


class TestKernelDensityEntropyBase:
    """Tests for logarithm base conversion."""

    def test_base_2_bits(self):
        """Base 2 gives entropy in bits."""
        torch.manual_seed(42)
        samples = torch.randn(500, 2)
        h_nats = kernel_density_entropy(samples)
        h_bits = kernel_density_entropy(samples, base=2)
        expected_bits = h_nats / math.log(2)
        assert torch.isclose(h_bits, expected_bits, rtol=1e-5)


class TestKernelDensityEntropyBatching:
    """Tests for batched operations."""

    def test_batch_consistency(self):
        """Batched computation matches individual computations."""
        torch.manual_seed(42)
        s1 = torch.randn(500, 3)
        s2 = torch.randn(500, 3)
        s_batch = torch.stack([s1, s2])

        h1 = kernel_density_entropy(s1)
        h2 = kernel_density_entropy(s2)
        h_batch = kernel_density_entropy(s_batch)

        assert torch.isclose(h_batch[0], h1, rtol=1e-5)
        assert torch.isclose(h_batch[1], h2, rtol=1e-5)


class TestKernelDensityEntropyEdgeCases:
    """Tests for edge cases and error handling."""

    def test_1d_input_raises(self):
        """Raises error for 1D input."""
        samples = torch.randn(100)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            kernel_density_entropy(samples)

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            kernel_density_entropy([[1, 2], [3, 4]])

    def test_invalid_kernel_raises(self):
        """Invalid kernel raises error."""
        samples = torch.randn(100, 2)
        with pytest.raises(ValueError, match="kernel must be"):
            kernel_density_entropy(samples, kernel="laplacian")


class TestKernelDensityEntropyDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        samples = torch.randn(500, 2, dtype=torch.float32)
        result = kernel_density_entropy(samples)
        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        samples = torch.randn(500, 2, dtype=torch.float64)
        result = kernel_density_entropy(samples)
        assert result.dtype == torch.float64


class TestKernelDensityEntropyDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        samples = torch.randn(500, 2, device="cpu")
        result = kernel_density_entropy(samples)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        samples = torch.randn(500, 2, device="cuda")
        result = kernel_density_entropy(samples)
        assert result.device.type == "cuda"


class TestKernelDensityEntropyGradients:
    """Tests for gradient computation."""

    def test_gradients_exist(self):
        """Gradients flow through the computation."""
        torch.manual_seed(42)
        samples = torch.randn(100, 2, requires_grad=True)
        result = kernel_density_entropy(samples)
        result.backward()
        assert samples.grad is not None
        assert torch.all(torch.isfinite(samples.grad))

    def test_gradcheck(self):
        """Passes gradient check."""
        torch.manual_seed(42)
        samples = torch.randn(30, 2, dtype=torch.float64, requires_grad=True)

        def func(x):
            return kernel_density_entropy(x, bandwidth=0.5)

        assert torch.autograd.gradcheck(func, samples, eps=1e-4, atol=1e-3)


class TestKernelDensityEntropyReproducibility:
    """Tests for reproducibility."""

    def test_deterministic(self):
        """Same input gives same output."""
        samples = torch.randn(500, 3)
        h1 = kernel_density_entropy(samples)
        h2 = kernel_density_entropy(samples)
        assert torch.equal(h1, h2)

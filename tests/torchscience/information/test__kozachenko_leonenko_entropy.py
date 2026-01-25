"""Comprehensive tests for Kozachenko-Leonenko k-NN entropy estimator."""

import math

import pytest
import torch

from torchscience.information import kozachenko_leonenko_entropy


class TestKozachenkoLeonenkoBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Returns scalar for (n_samples, n_dims) input."""
        samples = torch.randn(1000, 2)
        result = kozachenko_leonenko_entropy(samples)
        assert result.shape == torch.Size([])

    def test_output_shape_1d_samples(self):
        """Returns scalar for (n_samples, 1) input."""
        samples = torch.randn(1000, 1)
        result = kozachenko_leonenko_entropy(samples)
        assert result.shape == torch.Size([])

    def test_output_shape_3d_batch(self):
        """Returns 1D tensor for batched input."""
        samples = torch.randn(5, 1000, 3)
        result = kozachenko_leonenko_entropy(samples)
        assert result.shape == torch.Size([5])

    def test_output_shape_4d_batch(self):
        """Returns 2D tensor for nested batch."""
        samples = torch.randn(3, 4, 500, 2)
        result = kozachenko_leonenko_entropy(samples)
        assert result.shape == torch.Size([3, 4])

    def test_output_is_finite(self):
        """Result is finite for normal inputs."""
        torch.manual_seed(42)
        samples = torch.randn(500, 3)
        result = kozachenko_leonenko_entropy(samples)
        assert torch.isfinite(result)


class TestKozachenkoLeonenkoCorrectness:
    """Numerical correctness tests."""

    def test_gaussian_1d_entropy(self):
        """1D Gaussian entropy: H = 0.5 * log(2 * pi * e * sigma^2)."""
        torch.manual_seed(42)
        sigma = 1.0
        n_samples = 5000
        samples = torch.randn(n_samples, 1) * sigma
        result = kozachenko_leonenko_entropy(samples, k=3)
        expected = 0.5 * math.log(2 * math.pi * math.e * sigma**2)
        # KL estimator should be within 10% for large samples
        assert torch.isclose(result, torch.tensor(expected), rtol=0.15), (
            f"Expected {expected}, got {result}"
        )

    def test_gaussian_2d_entropy(self):
        """2D Gaussian entropy: H = d/2 * log(2 * pi * e * sigma^2)."""
        torch.manual_seed(42)
        sigma = 1.0
        d = 2
        n_samples = 5000
        samples = torch.randn(n_samples, d) * sigma
        result = kozachenko_leonenko_entropy(samples, k=3)
        expected = d / 2.0 * math.log(2 * math.pi * math.e * sigma**2)
        assert torch.isclose(result, torch.tensor(expected), rtol=0.15), (
            f"Expected {expected}, got {result}"
        )

    def test_gaussian_different_sigma(self):
        """Larger sigma gives higher entropy."""
        torch.manual_seed(42)
        n_samples = 2000
        samples_small_sigma = torch.randn(n_samples, 1) * 0.5
        samples_large_sigma = torch.randn(n_samples, 1) * 2.0
        h_small = kozachenko_leonenko_entropy(samples_small_sigma, k=3)
        h_large = kozachenko_leonenko_entropy(samples_large_sigma, k=3)
        assert h_large > h_small, (
            f"Larger sigma should give higher entropy: {h_large} vs {h_small}"
        )

    def test_higher_dimension_higher_entropy(self):
        """Higher dimensional Gaussian has higher entropy."""
        torch.manual_seed(42)
        n_samples = 2000
        samples_1d = torch.randn(n_samples, 1)
        samples_3d = torch.randn(n_samples, 3)
        h_1d = kozachenko_leonenko_entropy(samples_1d, k=3)
        h_3d = kozachenko_leonenko_entropy(samples_3d, k=3)
        assert h_3d > h_1d, (
            f"Higher dim should give higher entropy: {h_3d} vs {h_1d}"
        )

    def test_uniform_entropy_1d(self):
        """1D uniform on [0,1] has entropy ~ 0 (log(1) = 0)."""
        torch.manual_seed(42)
        n_samples = 5000
        samples = torch.rand(n_samples, 1)
        result = kozachenko_leonenko_entropy(samples, k=3)
        # Uniform on [0,1] has differential entropy = log(1) = 0
        # KL estimator has negative bias for bounded distributions due to edge effects
        # The estimate should be close to 0, within reasonable tolerance
        assert torch.isclose(result, torch.tensor(0.0), atol=0.6), (
            f"Expected ~0, got {result}"
        )


class TestKozachenkoLeonenkoKValues:
    """Tests for different k values."""

    def test_k1_original_estimator(self):
        """k=1 is the original KL estimator."""
        torch.manual_seed(42)
        samples = torch.randn(1000, 2)
        result = kozachenko_leonenko_entropy(samples, k=1)
        assert torch.isfinite(result)

    def test_k3_common_choice(self):
        """k=3 is a common choice (Kraskov default)."""
        torch.manual_seed(42)
        samples = torch.randn(1000, 2)
        result = kozachenko_leonenko_entropy(samples, k=3)
        assert torch.isfinite(result)

    def test_k5_higher_k(self):
        """k=5 for lower variance."""
        torch.manual_seed(42)
        samples = torch.randn(1000, 2)
        result = kozachenko_leonenko_entropy(samples, k=5)
        assert torch.isfinite(result)

    def test_different_k_values_similar(self):
        """Different k values give similar results for large samples."""
        torch.manual_seed(42)
        samples = torch.randn(5000, 2)
        h_k1 = kozachenko_leonenko_entropy(samples, k=1)
        h_k3 = kozachenko_leonenko_entropy(samples, k=3)
        h_k5 = kozachenko_leonenko_entropy(samples, k=5)
        # All should be within 20% of each other for Gaussian
        assert torch.isclose(h_k1, h_k3, rtol=0.2)
        assert torch.isclose(h_k3, h_k5, rtol=0.2)


class TestKozachenkoLeonenkoBase:
    """Tests for logarithm base conversion."""

    def test_base_2_bits(self):
        """Base 2 gives entropy in bits."""
        torch.manual_seed(42)
        samples = torch.randn(1000, 2)
        h_nats = kozachenko_leonenko_entropy(samples)
        h_bits = kozachenko_leonenko_entropy(samples, base=2)
        expected_bits = h_nats / math.log(2)
        assert torch.isclose(h_bits, torch.tensor(expected_bits), rtol=1e-5)

    def test_base_10_dits(self):
        """Base 10 gives entropy in dits."""
        torch.manual_seed(42)
        samples = torch.randn(1000, 2)
        h_nats = kozachenko_leonenko_entropy(samples)
        h_dits = kozachenko_leonenko_entropy(samples, base=10)
        expected_dits = h_nats / math.log(10)
        assert torch.isclose(h_dits, torch.tensor(expected_dits), rtol=1e-5)

    def test_base_e_same_as_none(self):
        """Base e should be same as None (default)."""
        torch.manual_seed(42)
        samples = torch.randn(1000, 2)
        h_none = kozachenko_leonenko_entropy(samples, base=None)
        h_e = kozachenko_leonenko_entropy(samples, base=math.e)
        assert torch.isclose(h_none, h_e, rtol=1e-5)


class TestKozachenkoLeonenkoBatching:
    """Tests for batched operations."""

    def test_batch_consistency(self):
        """Batched computation matches individual computations."""
        torch.manual_seed(42)
        samples1 = torch.randn(1000, 3)
        samples2 = torch.randn(1000, 3)
        samples_batch = torch.stack([samples1, samples2])

        h1 = kozachenko_leonenko_entropy(samples1)
        h2 = kozachenko_leonenko_entropy(samples2)
        h_batch = kozachenko_leonenko_entropy(samples_batch)

        assert torch.isclose(h_batch[0], h1, rtol=1e-5)
        assert torch.isclose(h_batch[1], h2, rtol=1e-5)

    def test_multi_batch_dims(self):
        """Works with multiple batch dimensions."""
        torch.manual_seed(42)
        samples = torch.randn(2, 3, 500, 4)
        result = kozachenko_leonenko_entropy(samples)
        assert result.shape == torch.Size([2, 3])
        assert torch.all(torch.isfinite(result))


class TestKozachenkoLeonenkoEdgeCases:
    """Tests for edge cases and error handling."""

    def test_minimum_samples(self):
        """Works with minimum viable number of samples."""
        torch.manual_seed(42)
        # k=1, need at least 2 samples
        samples = torch.randn(10, 2)
        result = kozachenko_leonenko_entropy(samples, k=1)
        assert torch.isfinite(result)

    def test_k_equals_n_minus_1(self):
        """Works when k = n_samples - 1."""
        torch.manual_seed(42)
        n = 10
        samples = torch.randn(n, 2)
        result = kozachenko_leonenko_entropy(samples, k=n - 1)
        assert torch.isfinite(result)

    def test_k_too_large_raises(self):
        """Raises error when k >= n_samples."""
        samples = torch.randn(10, 2)
        with pytest.raises(ValueError, match="k must be less than n_samples"):
            kozachenko_leonenko_entropy(samples, k=10)

    def test_k_zero_raises(self):
        """Raises error when k=0."""
        samples = torch.randn(100, 2)
        with pytest.raises(ValueError, match="k must be a positive integer"):
            kozachenko_leonenko_entropy(samples, k=0)

    def test_k_negative_raises(self):
        """Raises error for negative k."""
        samples = torch.randn(100, 2)
        with pytest.raises(ValueError, match="k must be a positive integer"):
            kozachenko_leonenko_entropy(samples, k=-1)

    def test_1d_input_raises(self):
        """Raises error for 1D input (missing n_dims)."""
        samples = torch.randn(100)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            kozachenko_leonenko_entropy(samples)

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            kozachenko_leonenko_entropy([[1, 2], [3, 4]])

    def test_invalid_base_raises(self):
        """Raises error for invalid base values."""
        samples = torch.randn(100, 2)
        with pytest.raises(ValueError, match="base must be positive"):
            kozachenko_leonenko_entropy(samples, base=0)
        with pytest.raises(ValueError, match="base must be positive"):
            kozachenko_leonenko_entropy(samples, base=-1)
        with pytest.raises(ValueError, match="not equal to 1"):
            kozachenko_leonenko_entropy(samples, base=1)

    def test_duplicate_points(self):
        """Handles duplicate points (zero distances)."""
        # Create samples with some duplicates
        samples = torch.zeros(100, 2)
        samples[:50] = torch.randn(50, 2)
        samples[50:] = samples[:50]  # Duplicate first half
        result = kozachenko_leonenko_entropy(samples, k=1)
        assert torch.isfinite(result)


class TestKozachenkoLeonenkoDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        samples = torch.randn(500, 2, dtype=torch.float32)
        result = kozachenko_leonenko_entropy(samples)
        assert result.dtype == torch.float32
        assert torch.isfinite(result)

    def test_float64(self):
        """Works with float64."""
        samples = torch.randn(500, 2, dtype=torch.float64)
        result = kozachenko_leonenko_entropy(samples)
        assert result.dtype == torch.float64
        assert torch.isfinite(result)


class TestKozachenkoLeonenkoDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        samples = torch.randn(500, 2, device="cpu")
        result = kozachenko_leonenko_entropy(samples)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        samples = torch.randn(500, 2, device="cuda")
        result = kozachenko_leonenko_entropy(samples)
        assert result.device.type == "cuda"


class TestKozachenkoLeonenkoGradients:
    """Tests for gradient computation."""

    def test_gradients_exist(self):
        """Gradients flow through the computation."""
        torch.manual_seed(42)
        samples = torch.randn(100, 2, requires_grad=True)
        result = kozachenko_leonenko_entropy(samples)
        result.backward()
        assert samples.grad is not None
        assert torch.all(torch.isfinite(samples.grad))

    def test_gradcheck(self):
        """Passes gradient check."""
        torch.manual_seed(42)
        samples = torch.randn(50, 2, dtype=torch.float64, requires_grad=True)

        def func(x):
            return kozachenko_leonenko_entropy(x, k=3)

        # Use larger eps for numerical stability
        assert torch.autograd.gradcheck(func, samples, eps=1e-4, atol=1e-3)


class TestKozachenkoLeonenkoReproducibility:
    """Tests for reproducibility."""

    def test_deterministic(self):
        """Same input gives same output."""
        samples = torch.randn(500, 3)
        h1 = kozachenko_leonenko_entropy(samples)
        h2 = kozachenko_leonenko_entropy(samples)
        assert torch.equal(h1, h2)

    def test_seed_reproducibility(self):
        """Results are reproducible with seed."""
        torch.manual_seed(42)
        samples1 = torch.randn(500, 3)
        h1 = kozachenko_leonenko_entropy(samples1)

        torch.manual_seed(42)
        samples2 = torch.randn(500, 3)
        h2 = kozachenko_leonenko_entropy(samples2)

        assert torch.equal(h1, h2)

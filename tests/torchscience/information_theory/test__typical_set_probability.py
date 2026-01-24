"""Tests for typical set probability."""

import math

import pytest
import torch

from torchscience.information import typical_set_probability


class TestTypicalSetProbabilityBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D input."""
        p = torch.tensor([0.5, 0.5])
        result = typical_set_probability(p, n=100, epsilon=0.1)
        assert result.shape == torch.Size([])

    def test_output_shape_2d(self):
        """Returns 1D for batched input."""
        p = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
        result = typical_set_probability(p, n=100, epsilon=0.1)
        assert result.shape == torch.Size([2])


class TestTypicalSetProbabilityCorrectness:
    """Numerical correctness tests."""

    def test_increases_with_n(self):
        """Probability bound increases with sequence length."""
        # Use non-uniform distribution (uniform has zero variance)
        p = torch.tensor([0.3, 0.7])
        epsilon = 0.1

        r10 = typical_set_probability(p, n=10, epsilon=epsilon)
        r100 = typical_set_probability(p, n=100, epsilon=epsilon)
        r1000 = typical_set_probability(p, n=1000, epsilon=epsilon)

        assert r100 > r10
        assert r1000 > r100

    def test_increases_with_epsilon(self):
        """Probability bound increases with larger epsilon."""
        # Use non-uniform distribution (uniform has zero variance)
        p = torch.tensor([0.3, 0.7])
        n = 100

        r_small = typical_set_probability(p, n=n, epsilon=0.05)
        r_large = typical_set_probability(p, n=n, epsilon=0.2)

        assert r_large > r_small

    def test_converges_to_one(self):
        """Probability approaches 1 for large n."""
        # Use non-uniform distribution (uniform has zero variance already)
        p = torch.tensor([0.3, 0.7])
        result = typical_set_probability(p, n=10000, epsilon=0.1)
        assert result > 0.99

    def test_deterministic_source_zero_variance(self):
        """Deterministic source (one symbol) has variance 0."""
        p = torch.tensor([1.0])
        result = typical_set_probability(p, n=10, epsilon=0.1)
        # Variance of -log p(X) is 0 for deterministic source
        # So bound is 1 - 0/(n*ε²) = 1
        assert torch.isclose(result, torch.tensor(1.0))

    def test_uniform_distribution(self):
        """Uniform distribution has specific variance."""
        # For uniform p = [1/n, ..., 1/n]
        # -log p(X) = log(n) always
        # Variance = 0
        n_symbols = 4
        p = torch.full((n_symbols,), 1.0 / n_symbols)
        result = typical_set_probability(p, n=10, epsilon=0.1)
        # Variance is 0 for constant random variable
        assert torch.isclose(result, torch.tensor(1.0))

    def test_bounded_zero_one(self):
        """Result is always in [0, 1]."""
        p = torch.tensor([0.9, 0.1])
        # Small n, small epsilon could give negative bound before clamping
        result = typical_set_probability(p, n=1, epsilon=0.01)
        assert result >= 0.0
        assert result <= 1.0

    def test_binary_entropy_variance(self):
        """Verify variance calculation for binary source."""
        # For Bernoulli(p): Y = -log(p) w.p. p, Y = -log(1-p) w.p. 1-p
        # E[Y] = -p*log(p) - (1-p)*log(1-p) = H(p)
        # E[Y²] = p*log²(p) + (1-p)*log²(1-p)
        # Var[Y] = E[Y²] - H(p)²

        p_val = 0.3
        p = torch.tensor([p_val, 1 - p_val])

        # Manual variance calculation
        log_p = math.log(p_val)
        log_1mp = math.log(1 - p_val)
        h = -p_val * log_p - (1 - p_val) * log_1mp
        e_y_sq = p_val * log_p**2 + (1 - p_val) * log_1mp**2
        var_y = e_y_sq - h**2

        # Chebyshev bound
        n = 50
        epsilon = 0.2
        expected_bound = 1 - var_y / (n * epsilon**2)

        result = typical_set_probability(p, n=n, epsilon=epsilon)
        assert torch.isclose(result, torch.tensor(expected_bound), atol=1e-5)


class TestTypicalSetProbabilityBase:
    """Tests for different logarithm bases."""

    def test_base_2(self):
        """Works with base 2."""
        p = torch.tensor([0.5, 0.5])
        result = typical_set_probability(p, n=100, epsilon=0.1, base=2.0)
        assert result >= 0.0 and result <= 1.0

    def test_base_e(self):
        """Explicit base e matches default."""
        p = torch.tensor([0.5, 0.5])
        result_default = typical_set_probability(p, n=100, epsilon=0.1)
        result_e = typical_set_probability(p, n=100, epsilon=0.1, base=math.e)
        assert torch.isclose(result_default, result_e)


class TestTypicalSetProbabilityBatching:
    """Tests for batched operations."""

    def test_batch_consistency(self):
        """Batched computation matches individual computations."""
        p1 = torch.tensor([0.5, 0.5])
        p2 = torch.tensor([0.3, 0.7])
        p_batch = torch.stack([p1, p2])

        r1 = typical_set_probability(p1, n=50, epsilon=0.1)
        r2 = typical_set_probability(p2, n=50, epsilon=0.1)
        r_batch = typical_set_probability(p_batch, n=50, epsilon=0.1)

        assert torch.isclose(r_batch[0], r1)
        assert torch.isclose(r_batch[1], r2)

    def test_multi_batch_dims(self):
        """Works with multiple batch dimensions."""
        p = torch.rand(2, 3, 4)
        p = p / p.sum(dim=-1, keepdim=True)
        result = typical_set_probability(p, n=100, epsilon=0.1)
        assert result.shape == torch.Size([2, 3])


class TestTypicalSetProbabilityEdgeCases:
    """Tests for edge cases and error handling."""

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            typical_set_probability([0.5, 0.5], n=100, epsilon=0.1)

    def test_scalar_raises(self):
        """Raises error for scalar input."""
        p = torch.tensor(0.5)
        with pytest.raises(ValueError, match="at least 1 dimension"):
            typical_set_probability(p, n=100, epsilon=0.1)

    def test_invalid_n_raises(self):
        """Raises error for n < 1."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="n must be at least 1"):
            typical_set_probability(p, n=0, epsilon=0.1)

    def test_invalid_epsilon_raises(self):
        """Raises error for non-positive epsilon."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="epsilon must be positive"):
            typical_set_probability(p, n=100, epsilon=0.0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            typical_set_probability(p, n=100, epsilon=-0.1)

    def test_zero_probability(self):
        """Handles zero probabilities correctly."""
        p = torch.tensor([0.5, 0.5, 0.0])
        result = typical_set_probability(p, n=100, epsilon=0.1)
        assert torch.isfinite(result)


class TestTypicalSetProbabilityDtypes:
    """Tests for different data types."""

    def test_float32_input(self):
        """Works with float32 input."""
        p = torch.tensor([0.5, 0.5], dtype=torch.float32)
        result = typical_set_probability(p, n=100, epsilon=0.1)
        assert result.dtype == torch.float32

    def test_float64_input(self):
        """Works with float64 input."""
        p = torch.tensor([0.5, 0.5], dtype=torch.float64)
        result = typical_set_probability(p, n=100, epsilon=0.1)
        assert result.dtype == torch.float32  # Converted internally


class TestTypicalSetProbabilityDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        p = torch.tensor([0.5, 0.5], device="cpu")
        result = typical_set_probability(p, n=100, epsilon=0.1)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        p = torch.tensor([0.5, 0.5], device="cuda")
        result = typical_set_probability(p, n=100, epsilon=0.1)
        assert result.device.type == "cuda"


class TestTypicalSetProbabilityReproducibility:
    """Tests for reproducibility."""

    def test_deterministic(self):
        """Same input gives same output."""
        p = torch.tensor([0.4, 0.3, 0.3])
        r1 = typical_set_probability(p, n=100, epsilon=0.1)
        r2 = typical_set_probability(p, n=100, epsilon=0.1)
        assert torch.equal(r1, r2)

"""Tests for entropy rate."""

import math

import pytest
import torch

from torchscience.information import entropy_rate


class TestEntropyRateBasic:
    """Basic functionality tests."""

    def test_output_shape_2d(self):
        """Returns scalar for 2D input (single transition matrix)."""
        P = torch.eye(3)
        result = entropy_rate(P)
        assert result.shape == torch.Size([])

    def test_output_shape_3d(self):
        """Returns 1D for batched input."""
        P = torch.stack([torch.eye(3), torch.eye(3)])
        result = entropy_rate(P)
        assert result.shape == torch.Size([2])


class TestEntropyRateCorrectness:
    """Numerical correctness tests."""

    def test_deterministic_chain(self):
        """Identity matrix (deterministic chain) has entropy rate 0."""
        P = torch.eye(3)
        result = entropy_rate(P)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_uniform_transitions(self):
        """Uniform transitions have entropy rate = log(n)."""
        n = 4
        P = torch.full((n, n), 1.0 / n)
        result = entropy_rate(P)
        expected = math.log(n)  # nats
        assert torch.isclose(result, torch.tensor(expected), atol=1e-5)

    def test_uniform_transitions_bits(self):
        """Uniform transitions with base=2 gives log2(n)."""
        n = 4
        P = torch.full((n, n), 1.0 / n)
        result = entropy_rate(P, base=2.0)
        expected = math.log2(n)  # bits
        assert torch.isclose(result, torch.tensor(expected), atol=1e-5)

    def test_binary_symmetric(self):
        """Binary symmetric channel-like transition matrix."""
        p = 0.2  # transition probability
        P = torch.tensor([[1 - p, p], [p, 1 - p]])
        result = entropy_rate(P, base=2.0)

        # Stationary distribution is uniform [0.5, 0.5]
        # Entropy rate = H(p) = -p*log2(p) - (1-p)*log2(1-p)
        expected = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        assert torch.isclose(result, torch.tensor(expected), atol=1e-5)

    def test_asymmetric_binary(self):
        """Asymmetric binary chain."""
        # P = [[0.9, 0.1], [0.3, 0.7]]
        P = torch.tensor([[0.9, 0.1], [0.3, 0.7]])
        result = entropy_rate(P, base=2.0)

        # Stationary distribution: π = [0.75, 0.25]
        # (solving π = π @ P)
        pi = torch.tensor([0.75, 0.25])

        # H∞ = 0.75 * H([0.9, 0.1]) + 0.25 * H([0.3, 0.7])
        h1 = -0.9 * math.log2(0.9) - 0.1 * math.log2(0.1)
        h2 = -0.3 * math.log2(0.3) - 0.7 * math.log2(0.7)
        expected = 0.75 * h1 + 0.25 * h2

        assert torch.isclose(result, torch.tensor(expected), atol=1e-4)

    def test_non_negative(self):
        """Entropy rate is non-negative."""
        P = torch.rand(5, 5)
        P = P / P.sum(dim=-1, keepdim=True)  # Make row-stochastic
        result = entropy_rate(P)
        assert result >= -1e-6

    def test_bounded_by_max_entropy(self):
        """Entropy rate is bounded by max row entropy."""
        n = 4
        P = torch.rand(n, n)
        P = P / P.sum(dim=-1, keepdim=True)
        result = entropy_rate(P, base=2.0)
        # Cannot exceed log(n) = 2 bits for n=4
        assert result <= math.log2(n) + 1e-5


class TestEntropyRateStationaryDistribution:
    """Tests for providing stationary distribution."""

    def test_provided_vs_computed(self):
        """Provided stationary distribution matches computed."""
        P = torch.tensor([[0.9, 0.1], [0.3, 0.7]])

        result_computed = entropy_rate(P)

        # Provide the correct stationary distribution
        pi = torch.tensor([0.75, 0.25])
        result_provided = entropy_rate(P, stationary_distribution=pi)

        assert torch.isclose(result_computed, result_provided, atol=1e-5)

    def test_uniform_stationary(self):
        """Uniform stationary distribution for doubly-stochastic matrix."""
        # Doubly-stochastic matrix has uniform stationary distribution
        P = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        pi = torch.tensor([0.5, 0.5])
        result = entropy_rate(P, stationary_distribution=pi, base=2.0)
        # Each row has H = 1 bit, so H∞ = 1 bit
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)


class TestEntropyRateBatching:
    """Tests for batched operations."""

    def test_batch_consistency(self):
        """Batched computation matches individual computations."""
        P1 = torch.eye(3)
        P2 = torch.full((3, 3), 1.0 / 3)
        P_batch = torch.stack([P1, P2])

        r1 = entropy_rate(P1)
        r2 = entropy_rate(P2)
        r_batch = entropy_rate(P_batch)

        assert torch.isclose(r_batch[0], r1, atol=1e-6)
        assert torch.isclose(r_batch[1], r2, atol=1e-5)

    def test_multi_batch_dims(self):
        """Works with multiple batch dimensions."""
        P = torch.rand(2, 3, 4, 4)
        P = P / P.sum(dim=-1, keepdim=True)
        result = entropy_rate(P)
        assert result.shape == torch.Size([2, 3])


class TestEntropyRateEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_state(self):
        """Single state has entropy rate 0."""
        P = torch.tensor([[1.0]])
        result = entropy_rate(P)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            entropy_rate([[0.5, 0.5], [0.5, 0.5]])

    def test_1d_input_raises(self):
        """Raises error for 1D input."""
        P = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            entropy_rate(P)

    def test_non_square_raises(self):
        """Raises error for non-square matrix."""
        P = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
        with pytest.raises(ValueError, match="must be square"):
            entropy_rate(P)

    def test_mismatched_stationary_raises(self):
        """Raises error for mismatched stationary distribution size."""
        P = torch.eye(3)
        pi = torch.tensor([0.5, 0.5])  # Wrong size
        with pytest.raises(ValueError, match="must match n_states"):
            entropy_rate(P, stationary_distribution=pi)


class TestEntropyRateDtypes:
    """Tests for different data types."""

    def test_float32_input(self):
        """Works with float32 input."""
        P = torch.eye(3, dtype=torch.float32)
        result = entropy_rate(P)
        assert result.dtype == torch.float32

    def test_float64_input(self):
        """Works with float64 input."""
        P = torch.eye(3, dtype=torch.float64)
        result = entropy_rate(P)
        assert result.dtype == torch.float32  # Converted internally


class TestEntropyRateDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        P = torch.eye(3, device="cpu")
        result = entropy_rate(P)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        P = torch.eye(3, device="cuda")
        result = entropy_rate(P)
        assert result.device.type == "cuda"


class TestEntropyRateReproducibility:
    """Tests for reproducibility."""

    def test_deterministic(self):
        """Same input gives same output."""
        P = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        r1 = entropy_rate(P)
        r2 = entropy_rate(P)
        assert torch.equal(r1, r2)


class TestEntropyRateIntegration:
    """Integration tests."""

    def test_vs_row_entropies(self):
        """Entropy rate equals weighted average of row entropies."""
        from torchscience.information import shannon_entropy

        P = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
        result = entropy_rate(P, base=2.0)

        # Compute manually
        pi = torch.tensor([0.6666667, 0.3333333])  # Stationary distribution
        h0 = shannon_entropy(P[0], base=2.0)
        h1 = shannon_entropy(P[1], base=2.0)
        expected = pi[0] * h0 + pi[1] * h1

        assert torch.isclose(result, expected, atol=1e-4)

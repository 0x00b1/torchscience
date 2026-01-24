"""Tests for Shannon's source coding bound."""

import pytest
import torch

from torchscience.information import source_coding_bound


class TestSourceCodingBoundBasic:
    """Basic functionality tests."""

    def test_returns_tuple(self):
        """Returns tuple of (lower, upper) bounds."""
        entropy = torch.tensor(1.0)
        result = source_coding_bound(entropy)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_shape_scalar(self):
        """Returns scalars for scalar input."""
        entropy = torch.tensor(1.0)
        lower, upper = source_coding_bound(entropy)
        assert lower.shape == torch.Size([])
        assert upper.shape == torch.Size([])

    def test_output_shape_1d(self):
        """Returns same shape for batched input."""
        entropy = torch.tensor([1.0, 2.0, 0.5])
        lower, upper = source_coding_bound(entropy)
        assert lower.shape == torch.Size([3])
        assert upper.shape == torch.Size([3])


class TestSourceCodingBoundCorrectness:
    """Numerical correctness tests."""

    def test_n_equals_1(self):
        """For n=1, bounds are [H, H+1]."""
        entropy = torch.tensor(1.0)
        lower, upper = source_coding_bound(entropy, n=1)
        assert torch.isclose(lower, torch.tensor(1.0))
        assert torch.isclose(upper, torch.tensor(2.0))

    def test_n_equals_10(self):
        """For n=10, bounds are [10*H, 10*H+1]."""
        entropy = torch.tensor(1.0)
        lower, upper = source_coding_bound(entropy, n=10)
        assert torch.isclose(lower, torch.tensor(10.0))
        assert torch.isclose(upper, torch.tensor(11.0))

    def test_zero_entropy(self):
        """Zero entropy gives [0, 1]."""
        entropy = torch.tensor(0.0)
        lower, upper = source_coding_bound(entropy, n=1)
        assert torch.isclose(lower, torch.tensor(0.0))
        assert torch.isclose(upper, torch.tensor(1.0))

    def test_upper_minus_lower_is_one(self):
        """Upper - lower = 1 always."""
        entropy = torch.tensor(2.5)
        for n in [1, 5, 10, 100]:
            lower, upper = source_coding_bound(entropy, n=n)
            assert torch.isclose(upper - lower, torch.tensor(1.0))

    def test_lower_bound_scales_with_n(self):
        """Lower bound scales linearly with n."""
        entropy = torch.tensor(2.0)
        lower_1, _ = source_coding_bound(entropy, n=1)
        lower_5, _ = source_coding_bound(entropy, n=5)
        assert torch.isclose(lower_5, 5 * lower_1)

    def test_per_symbol_overhead_decreases(self):
        """Per-symbol overhead (1/n) decreases with n."""
        entropy = torch.tensor(1.5)

        # Per-symbol upper bound is H + 1/n
        _, upper_1 = source_coding_bound(entropy, n=1)
        _, upper_10 = source_coding_bound(entropy, n=10)
        _, upper_100 = source_coding_bound(entropy, n=100)

        per_symbol_1 = upper_1 / 1
        per_symbol_10 = upper_10 / 10
        per_symbol_100 = upper_100 / 100

        assert per_symbol_1 > per_symbol_10
        assert per_symbol_10 > per_symbol_100


class TestSourceCodingBoundBatching:
    """Tests for batched operations."""

    def test_batch_consistency(self):
        """Batched computation matches individual computations."""
        h1 = torch.tensor(1.0)
        h2 = torch.tensor(2.0)
        h_batch = torch.stack([h1, h2])

        l1, u1 = source_coding_bound(h1, n=5)
        l2, u2 = source_coding_bound(h2, n=5)
        l_batch, u_batch = source_coding_bound(h_batch, n=5)

        assert torch.isclose(l_batch[0], l1)
        assert torch.isclose(l_batch[1], l2)
        assert torch.isclose(u_batch[0], u1)
        assert torch.isclose(u_batch[1], u2)

    def test_multi_batch_dims(self):
        """Works with multiple batch dimensions."""
        entropy = torch.rand(2, 3)
        lower, upper = source_coding_bound(entropy, n=5)
        assert lower.shape == torch.Size([2, 3])
        assert upper.shape == torch.Size([2, 3])


class TestSourceCodingBoundEdgeCases:
    """Tests for edge cases and error handling."""

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            source_coding_bound(1.0)

    def test_invalid_n_raises(self):
        """Raises error for n < 1."""
        entropy = torch.tensor(1.0)
        with pytest.raises(ValueError, match="n must be at least 1"):
            source_coding_bound(entropy, n=0)

    def test_large_n(self):
        """Works with large n."""
        entropy = torch.tensor(1.0)
        lower, upper = source_coding_bound(entropy, n=10000)
        assert torch.isclose(lower, torch.tensor(10000.0))
        assert torch.isclose(upper, torch.tensor(10001.0))


class TestSourceCodingBoundDtypes:
    """Tests for different data types."""

    def test_float32_input(self):
        """Works with float32 input."""
        entropy = torch.tensor(1.0, dtype=torch.float32)
        lower, upper = source_coding_bound(entropy)
        assert lower.dtype == torch.float32
        assert upper.dtype == torch.float32

    def test_float64_input(self):
        """Works with float64 input."""
        entropy = torch.tensor(1.0, dtype=torch.float64)
        lower, upper = source_coding_bound(entropy)
        # Note: implementation converts to float32
        assert lower.dtype == torch.float32


class TestSourceCodingBoundDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        entropy = torch.tensor(1.0, device="cpu")
        lower, upper = source_coding_bound(entropy)
        assert lower.device.type == "cpu"
        assert upper.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        entropy = torch.tensor(1.0, device="cuda")
        lower, upper = source_coding_bound(entropy)
        assert lower.device.type == "cuda"
        assert upper.device.type == "cuda"


class TestSourceCodingBoundReproducibility:
    """Tests for reproducibility."""

    def test_deterministic(self):
        """Same input gives same output."""
        entropy = torch.tensor(1.5)
        l1, u1 = source_coding_bound(entropy, n=7)
        l2, u2 = source_coding_bound(entropy, n=7)
        assert torch.equal(l1, l2)
        assert torch.equal(u1, u2)


class TestSourceCodingBoundIntegration:
    """Integration tests with other operators."""

    def test_huffman_satisfies_upper_bound(self):
        """Huffman average length < entropy + 1."""
        from torchscience.information import huffman_lengths, shannon_entropy

        probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
        entropy = shannon_entropy(probs, base=2.0)
        lengths = huffman_lengths(probs)
        avg_length = (probs * lengths).sum()

        lower, upper = source_coding_bound(entropy, n=1, base=2.0)

        # Huffman average length should be between bounds
        assert avg_length >= lower - 1e-6
        assert avg_length < upper + 1e-6

    def test_asymptotic_efficiency(self):
        """As n increases, per-symbol overhead approaches 0."""
        entropy = torch.tensor(1.5)

        # Per-symbol overhead at n=1 vs n=1000
        _, u1 = source_coding_bound(entropy, n=1)
        _, u1000 = source_coding_bound(entropy, n=1000)

        overhead_1 = (u1 - entropy) / 1
        overhead_1000 = (u1000 - 1000 * entropy) / 1000

        assert overhead_1000 < overhead_1 / 100

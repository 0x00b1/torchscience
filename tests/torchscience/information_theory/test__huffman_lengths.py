"""Tests for Huffman code lengths."""

import pytest
import torch

from torchscience.information import huffman_lengths


class TestHuffmanLengthsBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns same shape as input for 1D."""
        probs = torch.tensor([0.5, 0.25, 0.25])
        result = huffman_lengths(probs)
        assert result.shape == probs.shape

    def test_output_shape_2d_batch(self):
        """Returns same shape for batched input."""
        probs = torch.tensor([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]])
        result = huffman_lengths(probs)
        assert result.shape == probs.shape


class TestHuffmanLengthsCorrectness:
    """Numerical correctness tests."""

    def test_dyadic_distribution(self):
        """Dyadic distribution [0.5, 0.25, 0.25] gives lengths [1, 2, 2]."""
        probs = torch.tensor([0.5, 0.25, 0.25])
        result = huffman_lengths(probs)
        expected = torch.tensor([1.0, 2.0, 2.0])
        assert torch.equal(result, expected)

    def test_uniform_4_symbols(self):
        """Uniform distribution over 4 symbols gives all length 2."""
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        result = huffman_lengths(probs)
        expected = torch.tensor([2.0, 2.0, 2.0, 2.0])
        assert torch.equal(result, expected)

    def test_uniform_2_symbols(self):
        """Two equal symbols get length 1 each."""
        probs = torch.tensor([0.5, 0.5])
        result = huffman_lengths(probs)
        expected = torch.tensor([1.0, 1.0])
        assert torch.equal(result, expected)

    def test_uniform_8_symbols(self):
        """Uniform distribution over 8 symbols gives all length 3."""
        probs = torch.full((8,), 0.125)
        result = huffman_lengths(probs)
        expected = torch.full((8,), 3.0)
        assert torch.equal(result, expected)

    def test_highly_skewed(self):
        """Highly skewed distribution gives short length for frequent symbol."""
        probs = torch.tensor([0.9, 0.05, 0.05])
        result = huffman_lengths(probs)
        # Most probable symbol should have shortest code
        assert result[0] == torch.min(result)

    def test_kraft_inequality_satisfied(self):
        """Huffman lengths satisfy Kraft inequality with equality."""
        from torchscience.information import kraft_inequality

        probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
        lengths = huffman_lengths(probs)
        kraft_sum = kraft_inequality(lengths)
        assert torch.isclose(kraft_sum, torch.tensor(1.0))

    def test_shannon_lower_bound(self):
        """Average length >= entropy (Shannon lower bound)."""
        from torchscience.information import shannon_entropy

        probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
        lengths = huffman_lengths(probs)
        avg_length = (probs * lengths).sum()
        entropy = shannon_entropy(probs, base=2.0)
        assert avg_length >= entropy - 1e-6  # Small tolerance

    def test_shannon_upper_bound(self):
        """Average length < entropy + 1 (Shannon upper bound)."""
        from torchscience.information import shannon_entropy

        probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
        lengths = huffman_lengths(probs)
        avg_length = (probs * lengths).sum()
        entropy = shannon_entropy(probs, base=2.0)
        assert avg_length < entropy + 1.0

    def test_single_dominant_symbol(self):
        """One very dominant symbol gets shortest code."""
        probs = torch.tensor([0.99, 0.005, 0.005])
        result = huffman_lengths(probs)
        assert result[0] == torch.min(result)


class TestHuffmanLengthsBatching:
    """Tests for batched operations."""

    def test_batch_consistency(self):
        """Batched computation matches individual computations."""
        p1 = torch.tensor([0.5, 0.25, 0.25])
        p2 = torch.tensor([0.25, 0.25, 0.5])
        p_batch = torch.stack([p1, p2])

        r1 = huffman_lengths(p1)
        r2 = huffman_lengths(p2)
        r_batch = huffman_lengths(p_batch)

        assert torch.equal(r_batch[0], r1)
        assert torch.equal(r_batch[1], r2)

    def test_multi_batch_dims(self):
        """Works with multiple batch dimensions."""
        probs = torch.rand(2, 3, 4)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # Normalize
        result = huffman_lengths(probs)
        assert result.shape == torch.Size([2, 3, 4])


class TestHuffmanLengthsEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_symbol(self):
        """Single symbol gets length 0."""
        probs = torch.tensor([1.0])
        result = huffman_lengths(probs)
        assert torch.equal(result, torch.tensor([0.0]))

    def test_two_symbols_unequal(self):
        """Two unequal symbols still get length 1 each."""
        probs = torch.tensor([0.9, 0.1])
        result = huffman_lengths(probs)
        expected = torch.tensor([1.0, 1.0])
        assert torch.equal(result, expected)

    def test_empty_input(self):
        """Empty input returns empty output."""
        probs = torch.empty(0)
        result = huffman_lengths(probs)
        assert result.shape == torch.Size([0])

    def test_scalar_input_raises(self):
        """Raises error for scalar input."""
        probs = torch.tensor(0.5)
        with pytest.raises(ValueError, match="at least 1 dimension"):
            huffman_lengths(probs)

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            huffman_lengths([0.5, 0.25, 0.25])

    def test_alphabet_size_not_2_raises(self):
        """Raises NotImplementedError for non-binary alphabet."""
        probs = torch.tensor([0.5, 0.25, 0.25])
        with pytest.raises(NotImplementedError, match="Only binary"):
            huffman_lengths(probs, alphabet_size=3)

    def test_invalid_alphabet_size_raises(self):
        """Raises error for alphabet_size < 2."""
        probs = torch.tensor([0.5, 0.25, 0.25])
        with pytest.raises(
            ValueError, match="alphabet_size must be at least 2"
        ):
            huffman_lengths(probs, alphabet_size=1)


class TestHuffmanLengthsDtypes:
    """Tests for different data types."""

    def test_float32_input(self):
        """Works with float32 input."""
        probs = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32)
        result = huffman_lengths(probs)
        assert result.dtype == torch.float32

    def test_float64_input(self):
        """Works with float64 input."""
        probs = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
        result = huffman_lengths(probs)
        assert result.dtype == torch.float64


class TestHuffmanLengthsDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        probs = torch.tensor([0.5, 0.25, 0.25], device="cpu")
        result = huffman_lengths(probs)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        probs = torch.tensor([0.5, 0.25, 0.25], device="cuda")
        result = huffman_lengths(probs)
        assert result.device.type == "cuda"


class TestHuffmanLengthsReproducibility:
    """Tests for reproducibility."""

    def test_deterministic(self):
        """Same input gives same output."""
        probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
        r1 = huffman_lengths(probs)
        r2 = huffman_lengths(probs)
        assert torch.equal(r1, r2)


class TestHuffmanLengthsIntegration:
    """Integration tests with other operators."""

    def test_kraft_complete_code(self):
        """Huffman codes are complete (Kraft sum = 1)."""
        from torchscience.information import kraft_inequality

        for n in [3, 5, 8, 10]:
            probs = torch.rand(n)
            probs = probs / probs.sum()
            lengths = huffman_lengths(probs)
            kraft_sum = kraft_inequality(lengths)
            assert torch.isclose(kraft_sum, torch.tensor(1.0), atol=1e-6)

    def test_optimal_for_dyadic(self):
        """For dyadic distribution, lengths equal log(1/p)."""
        # Dyadic: probabilities are powers of 1/2
        probs = torch.tensor([0.5, 0.25, 0.125, 0.125])
        lengths = huffman_lengths(probs)
        # For dyadic, optimal lengths are -log2(p)
        expected_lengths = -torch.log2(probs)
        # Note: Huffman may swap equal-probability symbols
        assert torch.allclose(lengths.sort()[0], expected_lengths.sort()[0])

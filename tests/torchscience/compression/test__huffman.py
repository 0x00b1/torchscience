"""Tests for Huffman encoding."""

import pytest
import torch

from torchscience.compression import huffman_decode, huffman_encode


class TestHuffmanEncodeBasic:
    """Basic functionality tests for encoding."""

    def test_output_types(self):
        """Returns bitstream and codebook."""
        symbols = torch.tensor([0, 0, 1])
        bits, codebook = huffman_encode(symbols)
        assert isinstance(bits, list)
        assert isinstance(codebook, dict)

    def test_codebook_contains_symbols(self):
        """Codebook contains all unique symbols."""
        symbols = torch.tensor([0, 1, 2, 0, 1])
        _, codebook = huffman_encode(symbols)
        assert 0 in codebook
        assert 1 in codebook
        assert 2 in codebook


class TestHuffmanEncodeCorrectness:
    """Numerical correctness tests for encoding."""

    def test_frequent_symbol_shorter_code(self):
        """More frequent symbols get shorter codes."""
        # 0 appears 5 times, 1 appears 2 times, 2 appears 1 time
        symbols = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2])
        _, codebook = huffman_encode(symbols)
        # Symbol 0 should have shortest code
        assert len(codebook[0]) <= len(codebook[1])
        assert len(codebook[0]) <= len(codebook[2])

    def test_dyadic_distribution(self):
        """Dyadic distribution gives predictable lengths."""
        # Probabilities [0.5, 0.25, 0.25] should give lengths [1, 2, 2]
        symbols = torch.tensor([0, 0, 1, 2])  # frequencies match dyadic
        probs = torch.tensor([0.5, 0.25, 0.25])
        _, codebook = huffman_encode(symbols, probabilities=probs)
        assert len(codebook[0]) == 1
        assert len(codebook[1]) == 2
        assert len(codebook[2]) == 2

    def test_codes_are_prefix_free(self):
        """No code is prefix of another (prefix-free property)."""
        symbols = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        _, codebook = huffman_encode(symbols)
        codes = list(codebook.values())
        for i, code1 in enumerate(codes):
            for j, code2 in enumerate(codes):
                if i != j:
                    # Neither should be prefix of the other
                    min_len = min(len(code1), len(code2))
                    assert code1[:min_len] != code2[:min_len] or len(
                        code1
                    ) == len(code2)

    def test_single_symbol(self):
        """Single unique symbol gets code [0]."""
        symbols = torch.tensor([5, 5, 5, 5])
        _, codebook = huffman_encode(symbols)
        assert codebook[5] == [0]


class TestHuffmanDecodeBasic:
    """Basic functionality tests for decoding."""

    def test_output_type(self):
        """Returns tensor."""
        symbols = torch.tensor([0, 1, 0])
        bits, codebook = huffman_encode(symbols)
        result = huffman_decode(bits, codebook)
        assert isinstance(result, torch.Tensor)


class TestHuffmanRoundTrip:
    """Round-trip tests (encode then decode)."""

    def test_round_trip_simple(self):
        """Simple round trip."""
        symbols = torch.tensor([0, 0, 1, 2, 0, 1])
        bits, codebook = huffman_encode(symbols)
        decoded = huffman_decode(bits, codebook, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_single_symbol(self):
        """Round trip with single symbol type."""
        symbols = torch.tensor([3, 3, 3, 3, 3])
        bits, codebook = huffman_encode(symbols)
        decoded = huffman_decode(bits, codebook, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_many_symbols(self):
        """Round trip with many symbol types."""
        torch.manual_seed(42)
        symbols = torch.randint(0, 10, (50,))
        bits, codebook = huffman_encode(symbols)
        decoded = huffman_decode(bits, codebook, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_with_probabilities(self):
        """Round trip using explicit probabilities."""
        symbols = torch.tensor([0, 1, 2, 0, 0])
        probs = torch.tensor([0.5, 0.3, 0.2])
        bits, codebook = huffman_encode(symbols, probabilities=probs)
        decoded = huffman_decode(bits, codebook, length=len(symbols))
        assert torch.equal(decoded, symbols)


class TestHuffmanEdgeCases:
    """Edge case tests."""

    def test_empty_encode(self):
        """Empty input for encoding."""
        symbols = torch.empty(0, dtype=torch.long)
        bits, codebook = huffman_encode(symbols)
        assert bits == []
        assert codebook == {}

    def test_empty_decode(self):
        """Empty codebook for decoding."""
        result = huffman_decode([], {})
        assert result.numel() == 0

    def test_encode_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            huffman_encode([0, 1, 2])

    def test_encode_not_1d_raises(self):
        """Raises error for non-1D input."""
        symbols = torch.tensor([[0, 1], [2, 3]])
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            huffman_encode(symbols)

    def test_decode_partial(self):
        """Decode only part of bitstream."""
        symbols = torch.tensor([0, 0, 0, 1, 1])
        bits, codebook = huffman_encode(symbols)
        decoded = huffman_decode(bits, codebook, length=3)
        assert len(decoded) == 3
        assert torch.equal(decoded, symbols[:3])


class TestHuffmanCompression:
    """Compression effectiveness tests."""

    def test_compression_skewed_distribution(self):
        """Skewed distribution compresses better than uniform."""
        # Highly skewed: mostly 0s
        skewed = torch.tensor([0] * 80 + [1] * 15 + [2] * 5)
        bits_skewed, _ = huffman_encode(skewed)

        # Uniform-ish
        uniform = torch.tensor([0, 1, 2] * 33 + [0])
        bits_uniform, _ = huffman_encode(uniform)

        # Skewed should have fewer total bits
        assert len(bits_skewed) < len(bits_uniform)

    def test_kraft_inequality(self):
        """Code lengths satisfy Kraft inequality with equality."""
        symbols = torch.tensor([0, 0, 0, 1, 1, 2, 3, 4])
        _, codebook = huffman_encode(symbols)

        # Kraft sum = sum(2^{-length}) should equal 1 for complete code
        kraft_sum = sum(2 ** (-len(code)) for code in codebook.values())
        assert abs(kraft_sum - 1.0) < 1e-10

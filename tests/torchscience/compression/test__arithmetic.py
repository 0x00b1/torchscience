"""Tests for arithmetic coding."""

import math

import pytest
import torch

from torchscience.compression import arithmetic_decode, arithmetic_encode


class TestArithmeticEncodeBasic:
    """Basic functionality tests for encoding."""

    def test_output_types(self):
        """Returns bitstream and bit count."""
        symbols = torch.tensor([0, 1, 0])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        bits, n = arithmetic_encode(symbols, cdf)
        assert isinstance(bits, list)
        assert isinstance(n, int)

    def test_output_is_binary(self):
        """Bitstream contains only 0s and 1s."""
        symbols = torch.tensor([0, 1, 0, 1])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        bits, _ = arithmetic_encode(symbols, cdf)
        assert all(b in (0, 1) for b in bits)


class TestArithmeticRoundTrip:
    """Round-trip tests (encode then decode)."""

    def test_round_trip_uniform(self):
        """Round trip with uniform distribution."""
        symbols = torch.tensor([0, 1, 0, 1, 0])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        bits, _ = arithmetic_encode(symbols, cdf)
        decoded = arithmetic_decode(bits, cdf, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_skewed(self):
        """Round trip with skewed distribution."""
        symbols = torch.tensor([0, 0, 0, 1, 0, 0])
        cdf = torch.tensor([0.0, 0.8, 1.0])  # P(0)=0.8, P(1)=0.2
        bits, _ = arithmetic_encode(symbols, cdf)
        decoded = arithmetic_decode(bits, cdf, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_many_symbols(self):
        """Round trip with many symbol types."""
        symbols = torch.tensor([0, 1, 2, 3, 0, 1, 2])
        cdf = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        bits, _ = arithmetic_encode(symbols, cdf)
        decoded = arithmetic_decode(bits, cdf, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_single_symbol_type(self):
        """Round trip with single symbol type repeated."""
        symbols = torch.tensor([0, 0, 0, 0, 0])
        cdf = torch.tensor([0.0, 0.9, 1.0])
        bits, _ = arithmetic_encode(symbols, cdf)
        decoded = arithmetic_decode(bits, cdf, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_longer_sequence(self):
        """Round trip with longer sequence."""
        torch.manual_seed(42)
        symbols = torch.randint(0, 3, (20,))
        cdf = torch.tensor([0.0, 0.5, 0.8, 1.0])
        bits, _ = arithmetic_encode(symbols, cdf)
        decoded = arithmetic_decode(bits, cdf, length=len(symbols))
        assert torch.equal(decoded, symbols)


class TestArithmeticCompression:
    """Compression effectiveness tests."""

    def test_compression_approaches_entropy(self):
        """Compression rate approaches entropy."""
        # Highly skewed distribution: P(0) = 0.9, P(1) = 0.1
        # Entropy H = -0.9*log2(0.9) - 0.1*log2(0.1) ≈ 0.469 bits/symbol
        torch.manual_seed(42)
        n = 100
        # Generate symbols according to distribution
        symbols = (torch.rand(n) > 0.9).long()
        cdf = torch.tensor([0.0, 0.9, 1.0])

        bits, num_bits = arithmetic_encode(symbols, cdf)
        bits_per_symbol = num_bits / n

        # Entropy
        h = -0.9 * math.log2(0.9) - 0.1 * math.log2(0.1)

        # Should be within reasonable range of entropy
        # Arithmetic coding overhead is typically small
        assert bits_per_symbol < h + 0.5  # Allow some overhead

    def test_uniform_distribution_compression(self):
        """Uniform distribution gives ~log2(n) bits/symbol."""
        symbols = torch.tensor([0, 1, 2, 3] * 10)  # 40 symbols
        cdf = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        bits, num_bits = arithmetic_encode(symbols, cdf)
        bits_per_symbol = num_bits / len(symbols)

        # For uniform over 4 symbols, entropy = 2 bits
        assert 1.5 < bits_per_symbol < 2.5


class TestArithmeticEdgeCases:
    """Edge case tests."""

    def test_empty_encode(self):
        """Empty input for encoding."""
        symbols = torch.empty(0, dtype=torch.long)
        cdf = torch.tensor([0.0, 0.5, 1.0])
        bits, n = arithmetic_encode(symbols, cdf)
        assert bits == []
        assert n == 0

    def test_empty_decode(self):
        """Zero length decode."""
        cdf = torch.tensor([0.0, 0.5, 1.0])
        result = arithmetic_decode([], cdf, length=0)
        assert result.numel() == 0

    def test_encode_not_tensor_raises(self):
        """Raises error for non-tensor symbols."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            arithmetic_encode([0, 1], torch.tensor([0.0, 0.5, 1.0]))

    def test_encode_not_1d_raises(self):
        """Raises error for non-1D symbols."""
        symbols = torch.tensor([[0, 1], [1, 0]])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            arithmetic_encode(symbols, cdf)

    def test_symbol_out_of_range_raises(self):
        """Raises error for symbol out of range."""
        symbols = torch.tensor([0, 5, 0])  # 5 is out of range
        cdf = torch.tensor([0.0, 0.5, 1.0])  # Only 2 symbols
        with pytest.raises(ValueError, match="out of range"):
            arithmetic_encode(symbols, cdf)


class TestArithmeticPrecision:
    """Tests for different precision settings."""

    def test_lower_precision(self):
        """Works with lower precision."""
        symbols = torch.tensor([0, 1, 0, 1])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        bits, _ = arithmetic_encode(symbols, cdf, precision=16)
        decoded = arithmetic_decode(
            bits, cdf, length=len(symbols), precision=16
        )
        assert torch.equal(decoded, symbols)

    def test_higher_precision(self):
        """Works with higher precision."""
        symbols = torch.tensor([0, 1, 0, 1, 0, 0, 1])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        bits, _ = arithmetic_encode(symbols, cdf, precision=48)
        decoded = arithmetic_decode(
            bits, cdf, length=len(symbols), precision=48
        )
        assert torch.equal(decoded, symbols)

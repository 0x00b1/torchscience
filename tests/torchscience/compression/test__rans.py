"""Tests for rANS coding."""

import math

import pytest
import torch

from torchscience.compression import rans_decode, rans_encode


class TestRANSEncodeBasic:
    """Basic functionality tests for encoding."""

    def test_output_types(self):
        """Returns word list and word count."""
        symbols = torch.tensor([0, 1, 0])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        output, n = rans_encode(symbols, cdf)
        assert isinstance(output, list)
        assert isinstance(n, int)

    def test_output_is_16bit(self):
        """Output contains only 16-bit values."""
        symbols = torch.tensor([0, 1, 0, 1])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        output, _ = rans_encode(symbols, cdf)
        assert all(0 <= w <= 0xFFFF for w in output)


class TestRANSRoundTrip:
    """Round-trip tests (encode then decode)."""

    def test_round_trip_uniform(self):
        """Round trip with uniform distribution."""
        symbols = torch.tensor([0, 1, 0, 1, 0])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        output, _ = rans_encode(symbols, cdf)
        decoded = rans_decode(output, cdf, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_skewed(self):
        """Round trip with skewed distribution."""
        symbols = torch.tensor([0, 0, 0, 1, 0, 0])
        cdf = torch.tensor([0.0, 0.8, 1.0])  # P(0)=0.8, P(1)=0.2
        output, _ = rans_encode(symbols, cdf)
        decoded = rans_decode(output, cdf, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_many_symbols(self):
        """Round trip with many symbol types."""
        symbols = torch.tensor([0, 1, 2, 3, 0, 1, 2])
        cdf = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        output, _ = rans_encode(symbols, cdf)
        decoded = rans_decode(output, cdf, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_single_symbol_type(self):
        """Round trip with single symbol type repeated."""
        symbols = torch.tensor([0, 0, 0, 0, 0])
        cdf = torch.tensor([0.0, 0.9, 1.0])
        output, _ = rans_encode(symbols, cdf)
        decoded = rans_decode(output, cdf, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_longer_sequence(self):
        """Round trip with longer sequence."""
        torch.manual_seed(42)
        symbols = torch.randint(0, 3, (20,))
        cdf = torch.tensor([0.0, 0.5, 0.8, 1.0])
        output, _ = rans_encode(symbols, cdf)
        decoded = rans_decode(output, cdf, length=len(symbols))
        assert torch.equal(decoded, symbols)

    def test_round_trip_very_long_sequence(self):
        """Round trip with very long sequence."""
        torch.manual_seed(123)
        symbols = torch.randint(0, 4, (100,))
        cdf = torch.tensor([0.0, 0.1, 0.4, 0.7, 1.0])
        output, _ = rans_encode(symbols, cdf)
        decoded = rans_decode(output, cdf, length=len(symbols))
        assert torch.equal(decoded, symbols)


class TestRANSCompression:
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

        output, num_words = rans_encode(symbols, cdf)
        bits_per_symbol = (num_words * 16) / n

        # Entropy
        h = -0.9 * math.log2(0.9) - 0.1 * math.log2(0.1)

        # Should be within reasonable range of entropy
        # rANS has some overhead from state storage
        assert bits_per_symbol < h + 1.0

    def test_uniform_distribution_compression(self):
        """Uniform distribution gives ~log2(n) bits/symbol."""
        symbols = torch.tensor([0, 1, 2, 3] * 10)  # 40 symbols
        cdf = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        output, num_words = rans_encode(symbols, cdf)
        bits_per_symbol = (num_words * 16) / len(symbols)

        # For uniform over 4 symbols, entropy = 2 bits
        assert 1.5 < bits_per_symbol < 3.5


class TestRANSEdgeCases:
    """Edge case tests."""

    def test_empty_encode(self):
        """Empty input for encoding."""
        symbols = torch.empty(0, dtype=torch.long)
        cdf = torch.tensor([0.0, 0.5, 1.0])
        output, n = rans_encode(symbols, cdf)
        assert output == []
        assert n == 0

    def test_empty_decode(self):
        """Zero length decode."""
        cdf = torch.tensor([0.0, 0.5, 1.0])
        result = rans_decode([], cdf, length=0)
        assert result.numel() == 0

    def test_encode_not_tensor_raises(self):
        """Raises error for non-tensor symbols."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            rans_encode([0, 1], torch.tensor([0.0, 0.5, 1.0]))

    def test_encode_not_1d_raises(self):
        """Raises error for non-1D symbols."""
        symbols = torch.tensor([[0, 1], [1, 0]])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            rans_encode(symbols, cdf)

    def test_symbol_out_of_range_raises(self):
        """Raises error for symbol out of range."""
        symbols = torch.tensor([0, 5, 0])  # 5 is out of range
        cdf = torch.tensor([0.0, 0.5, 1.0])  # Only 2 symbols
        with pytest.raises(ValueError, match="out of range"):
            rans_encode(symbols, cdf)


class TestRANSPrecision:
    """Tests for different precision settings."""

    def test_lower_precision(self):
        """Works with lower precision."""
        symbols = torch.tensor([0, 1, 0, 1])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        output, _ = rans_encode(symbols, cdf, precision=12)
        decoded = rans_decode(output, cdf, length=len(symbols), precision=12)
        assert torch.equal(decoded, symbols)

    def test_standard_precision(self):
        """Works with standard precision."""
        symbols = torch.tensor([0, 1, 0, 1, 0, 0, 1])
        cdf = torch.tensor([0.0, 0.5, 1.0])
        output, _ = rans_encode(symbols, cdf, precision=16)
        decoded = rans_decode(output, cdf, length=len(symbols), precision=16)
        assert torch.equal(decoded, symbols)

"""Tests for LZ77 compression."""

import pytest
import torch

from torchscience.information.compression import lz77_decode, lz77_encode


class TestLZ77EncodeBasic:
    """Basic functionality tests for encoding."""

    def test_output_type(self):
        """Returns list of tuples."""
        symbols = torch.tensor([1, 2, 3])
        tokens = lz77_encode(symbols)
        assert isinstance(tokens, list)
        assert all(isinstance(t, tuple) and len(t) == 3 for t in tokens)

    def test_token_structure(self):
        """Each token is (offset, length, symbol)."""
        symbols = torch.tensor([1, 2, 1, 2])
        tokens = lz77_encode(symbols)
        for offset, length, symbol in tokens:
            assert isinstance(offset, int)
            assert isinstance(length, int)
            assert offset >= 0
            assert length >= 0


class TestLZ77EncodeCorrectness:
    """Correctness tests for encoding."""

    def test_finds_repetition(self):
        """Finds repeated patterns."""
        symbols = torch.tensor([1, 2, 3, 1, 2, 3])
        tokens = lz77_encode(symbols)
        # Should have at least one match (offset > 0, length > 0)
        has_match = any(t[0] > 0 and t[1] > 0 for t in tokens)
        assert has_match

    def test_no_match_for_unique(self):
        """Unique data has no matches."""
        symbols = torch.tensor([1, 2, 3, 4, 5])
        tokens = lz77_encode(symbols)
        # All should be literals (offset=0, length=0)
        for offset, length, _ in tokens:
            assert offset == 0 and length == 0


class TestLZ77DecodeBasic:
    """Basic functionality tests for decoding."""

    def test_output_type(self):
        """Returns tensor."""
        tokens = [(0, 0, 1), (0, 0, 2), (0, 0, 3)]
        result = lz77_decode(tokens)
        assert isinstance(result, torch.Tensor)


class TestLZ77RoundTrip:
    """Round-trip tests (encode then decode)."""

    def test_round_trip_simple(self):
        """Simple round trip."""
        symbols = torch.tensor([1, 2, 3, 1, 2, 3, 4])
        tokens = lz77_encode(symbols)
        decoded = lz77_decode(tokens)
        assert torch.equal(decoded, symbols)

    def test_round_trip_repeated_pattern(self):
        """Round trip with repeated pattern."""
        symbols = torch.tensor([1, 2] * 10)
        tokens = lz77_encode(symbols)
        decoded = lz77_decode(tokens)
        assert torch.equal(decoded, symbols)

    def test_round_trip_all_same(self):
        """Round trip with all same values."""
        symbols = torch.tensor([5] * 20)
        tokens = lz77_encode(symbols)
        decoded = lz77_decode(tokens)
        assert torch.equal(decoded, symbols)

    def test_round_trip_random(self):
        """Round trip with random data."""
        torch.manual_seed(42)
        symbols = torch.randint(0, 10, (50,))
        tokens = lz77_encode(symbols)
        decoded = lz77_decode(tokens)
        assert torch.equal(decoded, symbols)

    def test_round_trip_long_repeat(self):
        """Round trip with long repeated sequence."""
        # abcabcabcabc...
        pattern = torch.tensor([1, 2, 3])
        symbols = pattern.repeat(10)
        tokens = lz77_encode(symbols)
        decoded = lz77_decode(tokens)
        assert torch.equal(decoded, symbols)


class TestLZ77EdgeCases:
    """Edge case tests."""

    def test_empty_encode(self):
        """Empty input for encoding."""
        symbols = torch.empty(0, dtype=torch.long)
        tokens = lz77_encode(symbols)
        assert tokens == []

    def test_empty_decode(self):
        """Empty tokens for decoding."""
        result = lz77_decode([])
        assert result.numel() == 0

    def test_single_element(self):
        """Single element."""
        symbols = torch.tensor([7])
        tokens = lz77_encode(symbols)
        decoded = lz77_decode(tokens)
        assert torch.equal(decoded, symbols)

    def test_encode_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            lz77_encode([1, 2, 3])

    def test_encode_not_1d_raises(self):
        """Raises error for non-1D input."""
        symbols = torch.tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            lz77_encode(symbols)


class TestLZ77Compression:
    """Compression effectiveness tests."""

    def test_compresses_repetitive_data(self):
        """Repetitive data compresses well."""
        # abcabc... should compress significantly
        pattern = torch.tensor([1, 2, 3])
        symbols = pattern.repeat(20)  # 60 symbols
        tokens = lz77_encode(symbols)
        # Should have fewer tokens than symbols
        assert len(tokens) < len(symbols)

    def test_random_data_not_much_compression(self):
        """Random data doesn't compress much."""
        torch.manual_seed(42)
        # Truly random data
        symbols = torch.randint(0, 256, (100,))
        tokens = lz77_encode(symbols)
        # Tokens should be close to number of symbols
        # (no significant compression for random data)
        assert len(tokens) >= len(symbols) * 0.5  # Not much compression

    def test_window_size_affects_compression(self):
        """Larger window can find more matches."""
        # Create data with matches beyond small window
        pattern = torch.tensor([1, 2, 3, 4, 5])
        filler = torch.arange(100, 150)
        symbols = torch.cat([pattern, filler, pattern])

        tokens_small = lz77_encode(symbols, window_size=10)
        tokens_large = lz77_encode(symbols, window_size=100)

        # Larger window should find the distant match
        # resulting in fewer or equal tokens
        assert len(tokens_large) <= len(tokens_small)


class TestLZ77Parameters:
    """Tests for encoding parameters."""

    def test_custom_window_size(self):
        """Custom window size works."""
        symbols = torch.tensor([1, 2, 3, 1, 2, 3])
        tokens = lz77_encode(symbols, window_size=100)
        decoded = lz77_decode(tokens)
        assert torch.equal(decoded, symbols)

    def test_custom_lookahead_size(self):
        """Custom lookahead size works."""
        symbols = torch.tensor([1, 2, 3, 1, 2, 3])
        tokens = lz77_encode(symbols, lookahead_size=10)
        decoded = lz77_decode(tokens)
        assert torch.equal(decoded, symbols)

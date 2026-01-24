"""Tests for run-length encoding."""

import pytest
import torch

from torchscience.compression import run_length_decode, run_length_encode


class TestRunLengthEncodeBasic:
    """Basic functionality tests for encoding."""

    def test_output_types(self):
        """Returns tuple of two tensors."""
        symbols = torch.tensor([1, 1, 2])
        values, lengths = run_length_encode(symbols)
        assert isinstance(values, torch.Tensor)
        assert isinstance(lengths, torch.Tensor)

    def test_output_shapes(self):
        """Output shapes match number of runs."""
        symbols = torch.tensor([1, 1, 1, 2, 2, 3])
        values, lengths = run_length_encode(symbols)
        assert values.shape == torch.Size([3])
        assert lengths.shape == torch.Size([3])


class TestRunLengthEncodeCorrectness:
    """Numerical correctness tests for encoding."""

    def test_simple_runs(self):
        """Basic run-length encoding."""
        symbols = torch.tensor([1, 1, 1, 2, 2, 3])
        values, lengths = run_length_encode(symbols)
        assert torch.equal(values, torch.tensor([1, 2, 3]))
        assert torch.equal(lengths, torch.tensor([3, 2, 1]))

    def test_single_repeated_value(self):
        """Single value repeated."""
        symbols = torch.tensor([5, 5, 5, 5])
        values, lengths = run_length_encode(symbols)
        assert torch.equal(values, torch.tensor([5]))
        assert torch.equal(lengths, torch.tensor([4]))

    def test_alternating_values(self):
        """Alternating values (worst case)."""
        symbols = torch.tensor([1, 2, 1, 2])
        values, lengths = run_length_encode(symbols)
        assert torch.equal(values, torch.tensor([1, 2, 1, 2]))
        assert torch.equal(lengths, torch.tensor([1, 1, 1, 1]))

    def test_single_element(self):
        """Single element input."""
        symbols = torch.tensor([7])
        values, lengths = run_length_encode(symbols)
        assert torch.equal(values, torch.tensor([7]))
        assert torch.equal(lengths, torch.tensor([1]))

    def test_preserves_dtype(self):
        """Output values preserve input dtype."""
        symbols = torch.tensor([1.5, 1.5, 2.5], dtype=torch.float32)
        values, lengths = run_length_encode(symbols)
        assert values.dtype == torch.float32


class TestRunLengthDecodeBasic:
    """Basic functionality tests for decoding."""

    def test_output_type(self):
        """Returns tensor."""
        values = torch.tensor([1, 2])
        lengths = torch.tensor([2, 3])
        result = run_length_decode(values, lengths)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self):
        """Output length is sum of run lengths."""
        values = torch.tensor([1, 2, 3])
        lengths = torch.tensor([3, 2, 1])
        result = run_length_decode(values, lengths)
        assert result.shape == torch.Size([6])


class TestRunLengthDecodeCorrectness:
    """Numerical correctness tests for decoding."""

    def test_simple_decode(self):
        """Basic decoding."""
        values = torch.tensor([1, 2, 3])
        lengths = torch.tensor([3, 2, 1])
        result = run_length_decode(values, lengths)
        expected = torch.tensor([1, 1, 1, 2, 2, 3])
        assert torch.equal(result, expected)

    def test_single_run(self):
        """Single run decoding."""
        values = torch.tensor([5])
        lengths = torch.tensor([4])
        result = run_length_decode(values, lengths)
        expected = torch.tensor([5, 5, 5, 5])
        assert torch.equal(result, expected)


class TestRunLengthRoundTrip:
    """Round-trip tests (encode then decode)."""

    def test_round_trip_simple(self):
        """Simple round trip."""
        original = torch.tensor([1, 1, 1, 2, 2, 3])
        values, lengths = run_length_encode(original)
        decoded = run_length_decode(values, lengths)
        assert torch.equal(decoded, original)

    def test_round_trip_alternating(self):
        """Round trip with alternating values."""
        original = torch.tensor([1, 2, 1, 2, 1])
        values, lengths = run_length_encode(original)
        decoded = run_length_decode(values, lengths)
        assert torch.equal(decoded, original)

    def test_round_trip_single_value(self):
        """Round trip with repeated single value."""
        original = torch.tensor([42, 42, 42, 42, 42])
        values, lengths = run_length_encode(original)
        decoded = run_length_decode(values, lengths)
        assert torch.equal(decoded, original)

    def test_round_trip_random(self):
        """Round trip with random data."""
        torch.manual_seed(42)
        # Create data with some runs
        original = torch.randint(0, 5, (100,))
        values, lengths = run_length_encode(original)
        decoded = run_length_decode(values, lengths)
        assert torch.equal(decoded, original)


class TestRunLengthEdgeCases:
    """Edge case tests."""

    def test_empty_encode(self):
        """Empty input for encoding."""
        symbols = torch.empty(0, dtype=torch.long)
        values, lengths = run_length_encode(symbols)
        assert values.numel() == 0
        assert lengths.numel() == 0

    def test_empty_decode(self):
        """Empty input for decoding."""
        values = torch.empty(0, dtype=torch.long)
        lengths = torch.empty(0, dtype=torch.long)
        result = run_length_decode(values, lengths)
        assert result.numel() == 0

    def test_encode_not_tensor_raises(self):
        """Raises error for non-tensor input to encode."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            run_length_encode([1, 1, 2])

    def test_encode_not_1d_raises(self):
        """Raises error for non-1D input to encode."""
        symbols = torch.tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            run_length_encode(symbols)

    def test_decode_mismatched_lengths_raises(self):
        """Raises error if values and lengths have different sizes."""
        values = torch.tensor([1, 2, 3])
        lengths = torch.tensor([1, 2])
        with pytest.raises(ValueError, match="must have same length"):
            run_length_decode(values, lengths)


class TestRunLengthDevice:
    """Device handling tests."""

    def test_encode_cpu(self):
        """Encoding works on CPU."""
        symbols = torch.tensor([1, 1, 2], device="cpu")
        values, lengths = run_length_encode(symbols)
        assert values.device.type == "cpu"
        assert lengths.device.type == "cpu"

    def test_decode_cpu(self):
        """Decoding works on CPU."""
        values = torch.tensor([1, 2], device="cpu")
        lengths = torch.tensor([2, 1], device="cpu")
        result = run_length_decode(values, lengths)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_encode_cuda(self):
        """Encoding works on CUDA."""
        symbols = torch.tensor([1, 1, 2], device="cuda")
        values, lengths = run_length_encode(symbols)
        assert values.device.type == "cuda"
        assert lengths.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_decode_cuda(self):
        """Decoding works on CUDA."""
        values = torch.tensor([1, 2], device="cuda")
        lengths = torch.tensor([2, 1], device="cuda")
        result = run_length_decode(values, lengths)
        assert result.device.type == "cuda"


class TestRunLengthCompression:
    """Compression effectiveness tests."""

    def test_compresses_repetitive_data(self):
        """Repetitive data compresses well."""
        # 100 elements with runs
        symbols = torch.tensor([0] * 50 + [1] * 30 + [2] * 20)
        values, lengths = run_length_encode(symbols)
        # Should compress to 3 values
        assert len(values) == 3
        assert lengths.sum() == 100

    def test_worst_case_no_compression(self):
        """Alternating data doesn't compress."""
        symbols = torch.tensor([0, 1] * 50)
        values, lengths = run_length_encode(symbols)
        # No compression - 100 runs
        assert len(values) == 100

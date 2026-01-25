"""Tests for Kraft inequality."""

import pytest
import torch

from torchscience.information import kraft_inequality


class TestKraftInequalityBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D input."""
        lengths = torch.tensor([1, 2, 2])
        result = kraft_inequality(lengths)
        assert result.shape == torch.Size([])

    def test_output_shape_2d_batch(self):
        """Returns 1D tensor for batched input."""
        lengths = torch.tensor([[1, 2, 2], [2, 2, 2]])
        result = kraft_inequality(lengths)
        assert result.shape == torch.Size([2])


class TestKraftInequalityCorrectness:
    """Numerical correctness tests."""

    def test_optimal_binary_code(self):
        """Optimal binary code [1, 2, 2] has Kraft sum = 1."""
        lengths = torch.tensor([1, 2, 2])
        result = kraft_inequality(lengths)
        expected = 0.5 + 0.25 + 0.25  # 2^{-1} + 2^{-2} + 2^{-2}
        assert torch.isclose(result, torch.tensor(expected))

    def test_suboptimal_code(self):
        """Suboptimal code [2, 2, 2] has Kraft sum < 1."""
        lengths = torch.tensor([2, 2, 2])
        result = kraft_inequality(lengths)
        expected = 0.25 + 0.25 + 0.25  # 0.75
        assert torch.isclose(result, torch.tensor(expected))
        assert result < 1.0

    def test_invalid_code(self):
        """Invalid code [1, 1, 1] has Kraft sum > 1."""
        lengths = torch.tensor([1, 1, 1])
        result = kraft_inequality(lengths)
        expected = 0.5 + 0.5 + 0.5  # 1.5
        assert torch.isclose(result, torch.tensor(expected))
        assert result > 1.0

    def test_single_symbol(self):
        """Single symbol with length 0 has Kraft sum = 1."""
        lengths = torch.tensor([0])
        result = kraft_inequality(lengths)
        assert torch.isclose(result, torch.tensor(1.0))

    def test_uniform_lengths(self):
        """Uniform lengths for n symbols: n * D^{-l}."""
        n = 4
        length = 2
        lengths = torch.full((n,), length)
        result = kraft_inequality(lengths)
        expected = n * (2 ** (-length))  # 4 * 0.25 = 1.0
        assert torch.isclose(result, torch.tensor(expected))

    def test_huffman_like_dyadic(self):
        """Dyadic distribution [0.5, 0.25, 0.25] gives lengths [1, 2, 2]."""
        lengths = torch.tensor([1, 2, 2])
        result = kraft_inequality(lengths)
        assert torch.isclose(result, torch.tensor(1.0))

    def test_extended_code(self):
        """Longer code: [1, 2, 3, 3, 3, 3]."""
        lengths = torch.tensor([1, 2, 3, 3, 3, 3])
        result = kraft_inequality(lengths)
        # 0.5 + 0.25 + 4*0.125 = 0.5 + 0.25 + 0.5 = 1.25
        expected = 0.5 + 0.25 + 4 * 0.125
        assert torch.isclose(result, torch.tensor(expected))


class TestKraftInequalityAlphabetSize:
    """Tests for different alphabet sizes."""

    def test_ternary_code(self):
        """Ternary code (alphabet_size=3)."""
        lengths = torch.tensor([1, 1, 1])
        result = kraft_inequality(lengths, alphabet_size=3)
        expected = 3 * (1 / 3)  # 3^{-1} * 3 = 1.0
        assert torch.isclose(result, torch.tensor(expected))

    def test_quaternary_code(self):
        """Quaternary code (alphabet_size=4)."""
        lengths = torch.tensor([1, 1, 1, 1])
        result = kraft_inequality(lengths, alphabet_size=4)
        expected = 4 * (1 / 4)  # 1.0
        assert torch.isclose(result, torch.tensor(expected))

    def test_alphabet_size_2_explicit(self):
        """Explicit alphabet_size=2 matches default."""
        lengths = torch.tensor([1, 2, 2])
        default_result = kraft_inequality(lengths)
        explicit_result = kraft_inequality(lengths, alphabet_size=2)
        assert torch.equal(default_result, explicit_result)


class TestKraftInequalityBatching:
    """Tests for batched operations."""

    def test_batch_consistency(self):
        """Batched computation matches individual computations."""
        l1 = torch.tensor([1, 2, 2])
        l2 = torch.tensor([2, 2, 2])
        l_batch = torch.stack([l1, l2])

        r1 = kraft_inequality(l1)
        r2 = kraft_inequality(l2)
        r_batch = kraft_inequality(l_batch)

        assert torch.isclose(r_batch[0], r1)
        assert torch.isclose(r_batch[1], r2)

    def test_multi_batch_dims(self):
        """Works with multiple batch dimensions."""
        lengths = torch.randint(1, 5, (2, 3, 4))
        result = kraft_inequality(lengths)
        assert result.shape == torch.Size([2, 3])


class TestKraftInequalityEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_length(self):
        """Length 0 contributes D^0 = 1."""
        lengths = torch.tensor([0])
        result = kraft_inequality(lengths)
        assert torch.isclose(result, torch.tensor(1.0))

    def test_large_lengths(self):
        """Large lengths give small contributions."""
        lengths = torch.tensor([10, 10, 10])
        result = kraft_inequality(lengths)
        expected = 3 * (2 ** (-10))  # Very small
        assert torch.isclose(result, torch.tensor(expected))

    def test_scalar_input_raises(self):
        """Raises error for scalar input."""
        lengths = torch.tensor(2)
        with pytest.raises(ValueError, match="at least 1 dimension"):
            kraft_inequality(lengths)

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            kraft_inequality([1, 2, 2])

    def test_invalid_alphabet_size_raises(self):
        """Raises error for alphabet_size < 2."""
        lengths = torch.tensor([1, 2, 2])
        with pytest.raises(
            ValueError, match="alphabet_size must be at least 2"
        ):
            kraft_inequality(lengths, alphabet_size=1)


class TestKraftInequalityDtypes:
    """Tests for different data types."""

    def test_int_input(self):
        """Works with integer input."""
        lengths = torch.tensor([1, 2, 2], dtype=torch.int64)
        result = kraft_inequality(lengths)
        assert result.dtype == torch.float32

    def test_float_input(self):
        """Works with float input."""
        lengths = torch.tensor([1.0, 2.0, 2.0])
        result = kraft_inequality(lengths)
        assert torch.isclose(result, torch.tensor(1.0))


class TestKraftInequalityDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        lengths = torch.tensor([1, 2, 2], device="cpu")
        result = kraft_inequality(lengths)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        lengths = torch.tensor([1, 2, 2], device="cuda")
        result = kraft_inequality(lengths)
        assert result.device.type == "cuda"


class TestKraftInequalityGradients:
    """Tests for gradient computation."""

    def test_gradients_exist(self):
        """Gradients flow through the computation."""
        lengths = torch.tensor([1.0, 2.0, 2.0], requires_grad=True)
        result = kraft_inequality(lengths)
        result.backward()
        assert lengths.grad is not None


class TestKraftInequalityReproducibility:
    """Tests for reproducibility."""

    def test_deterministic(self):
        """Same input gives same output."""
        lengths = torch.tensor([1, 2, 3, 4])
        r1 = kraft_inequality(lengths)
        r2 = kraft_inequality(lengths)
        assert torch.equal(r1, r2)

"""Comprehensive tests for Tsallis entropy."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information_theory import shannon_entropy, tsallis_entropy


class TestTsallisEntropyBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D probability vector."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        result = tsallis_entropy(p, q=2)
        assert result.shape == torch.Size([])

    def test_output_shape_batch(self):
        """Returns correct shape for batched input."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        result = tsallis_entropy(p, q=2)
        assert result.shape == torch.Size([10])


class TestTsallisEntropyCorrectness:
    """Numerical correctness tests."""

    def test_uniform_distribution_q2(self):
        """Uniform distribution: S_2 = (n-1)/n for n outcomes."""
        n = 4
        p = torch.ones(n) / n
        result = tsallis_entropy(p, q=2)
        # S_2 = (1 - sum p_i^2) / (q-1) = (1 - n * (1/n)^2) / 1 = (1 - 1/n) = (n-1)/n
        expected = (n - 1) / n
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)

    def test_converges_to_shannon_from_below(self):
        """S_q -> H (Shannon) as q -> 1 from below."""
        torch.manual_seed(42)
        p = torch.softmax(torch.randn(10), dim=-1)
        h_shannon = shannon_entropy(p)

        for q in [0.9, 0.99, 0.999]:
            s_tsallis = tsallis_entropy(p, q=q)
            # Use looser tolerance for values further from 1
            rtol = 0.2 if abs(q - 1) > 0.05 else 0.1
            assert torch.isclose(s_tsallis, h_shannon, rtol=rtol)

    def test_converges_to_shannon_from_above(self):
        """S_q -> H (Shannon) as q -> 1 from above."""
        torch.manual_seed(42)
        p = torch.softmax(torch.randn(10), dim=-1)
        h_shannon = shannon_entropy(p)

        for q in [1.1, 1.01, 1.001]:
            s_tsallis = tsallis_entropy(p, q=q)
            # Use looser tolerance for values further from 1
            rtol = 0.2 if abs(q - 1) > 0.05 else 0.1
            assert torch.isclose(s_tsallis, h_shannon, rtol=rtol)

    def test_manual_calculation(self):
        """Verify against manual calculation."""
        p = torch.tensor([0.5, 0.3, 0.2])
        q = 2.0
        # S_q = (1 - sum p_i^q) / (q - 1)
        sum_p_q = 0.5**2 + 0.3**2 + 0.2**2  # = 0.25 + 0.09 + 0.04 = 0.38
        expected = (1 - sum_p_q) / (q - 1)  # = (1 - 0.38) / 1 = 0.62

        result = tsallis_entropy(p, q=q)
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)

    def test_delta_distribution(self):
        """Delta distribution (one outcome with prob 1) has zero entropy."""
        p = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = tsallis_entropy(p, q=2)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_relationship_to_renyi(self):
        """S_q = (1 - exp((1-q) H_q)) / (q-1) for Renyi entropy H_q."""
        from torchscience.information_theory import renyi_entropy

        p = torch.softmax(torch.randn(10), dim=-1)

        for q in [0.5, 2.0, 3.0]:
            s_tsallis = tsallis_entropy(p, q=q)
            h_renyi = renyi_entropy(p, alpha=q)

            # S_q = (1 - exp((1-q) * H_q)) / (q - 1)
            expected = (1 - torch.exp((1 - q) * h_renyi)) / (q - 1)

            assert torch.isclose(s_tsallis, expected, rtol=1e-4)


class TestTsallisEntropyGradients:
    """Gradient computation tests."""

    def test_gradcheck_q2(self):
        """Gradients are correct for q=2."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return tsallis_entropy(p_in, q=2.0)

        assert gradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_q0_5(self):
        """Gradients are correct for q=0.5."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return tsallis_entropy(p_in, q=0.5)

        assert gradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_batched(self):
        """Gradients are correct for batched input."""
        p = torch.softmax(torch.randn(3, 5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return tsallis_entropy(p_in, q=2.0, reduction="sum")

        assert gradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)


class TestTsallisEntropyInputTypes:
    """Input type handling tests."""

    def test_logits_input(self):
        """Correctly handles logits input."""
        logits = torch.randn(10)
        p = torch.softmax(logits, dim=-1)

        s_prob = tsallis_entropy(p, q=2, input_type="probability")
        s_logits = tsallis_entropy(logits, q=2, input_type="logits")

        assert torch.isclose(s_prob, s_logits, rtol=1e-5)

    def test_log_probability_input(self):
        """Correctly handles log probability input."""
        p = torch.softmax(torch.randn(10), dim=-1)
        log_p = torch.log(p)

        s_prob = tsallis_entropy(p, q=2, input_type="probability")
        s_log = tsallis_entropy(log_p, q=2, input_type="log_probability")

        assert torch.isclose(s_prob, s_log, rtol=1e-5)


class TestTsallisEntropyReductions:
    """Reduction mode tests."""

    def test_reduction_none(self):
        """None reduction returns per-sample values."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        result = tsallis_entropy(p, q=2, reduction="none")
        assert result.shape == torch.Size([10])

    def test_reduction_mean(self):
        """Mean reduction returns scalar."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        result_none = tsallis_entropy(p, q=2, reduction="none")
        result_mean = tsallis_entropy(p, q=2, reduction="mean")

        assert result_mean.shape == torch.Size([])
        assert torch.isclose(result_mean, result_none.mean(), rtol=1e-5)

    def test_reduction_sum(self):
        """Sum reduction returns scalar."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        result_none = tsallis_entropy(p, q=2, reduction="none")
        result_sum = tsallis_entropy(p, q=2, reduction="sum")

        assert result_sum.shape == torch.Size([])
        assert torch.isclose(result_sum, result_none.sum(), rtol=1e-5)


class TestTsallisEntropyValidation:
    """Input validation tests."""

    def test_invalid_q_one(self):
        """Raises error for q=1."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="q cannot be 1"):
            tsallis_entropy(p, q=1.0)

    def test_invalid_input_type(self):
        """Raises error for invalid input_type."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="input_type must be one of"):
            tsallis_entropy(p, q=2, input_type="invalid")

    def test_invalid_reduction(self):
        """Raises error for invalid reduction."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="reduction must be one of"):
            tsallis_entropy(p, q=2, reduction="invalid")

    def test_non_tensor_input(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="p must be a Tensor"):
            tsallis_entropy([0.5, 0.5], q=2)

    def test_invalid_dim(self):
        """Raises error for out of range dim."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(IndexError, match="dim .* out of range"):
            tsallis_entropy(p, q=2, dim=2)


class TestTsallisEntropyMeta:
    """Meta tensor support tests."""

    def test_meta_tensor_shape(self):
        """Meta tensors return correct shape."""
        p = torch.softmax(torch.randn(10, 5, device="meta"), dim=-1)
        result = tsallis_entropy(p, q=2)
        assert result.shape == torch.Size([10])
        assert result.device.type == "meta"


class TestTsallisEntropyDtypes:
    """Data type support tests."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Supports various floating point dtypes."""
        p = torch.softmax(torch.randn(10, dtype=dtype), dim=-1)
        result = tsallis_entropy(p, q=2)
        assert result.dtype == dtype

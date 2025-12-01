"""Comprehensive tests for Shannon entropy."""

import math

import pytest
import scipy.stats
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.information_theory import shannon_entropy


class TestShannonEntropyBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D probability vector."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        result = shannon_entropy(p)
        assert result.shape == torch.Size([])

    def test_output_shape_2d_batch(self):
        """Returns 1D tensor for batch of distributions."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        result = shannon_entropy(p)
        assert result.shape == torch.Size([10])

    def test_output_shape_3d_batch(self):
        """Returns 2D tensor for nested batch of distributions."""
        p = torch.softmax(torch.randn(4, 5, 8), dim=-1)
        result = shannon_entropy(p)
        assert result.shape == torch.Size([4, 5])

    def test_non_negativity(self):
        """Shannon entropy is always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(100), dim=-1)
            result = shannon_entropy(p)
            assert result >= 0, f"Entropy should be non-negative, got {result}"


class TestShannonEntropyCorrectness:
    """Numerical correctness tests."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution has maximum entropy = log(n)."""
        n = 8
        p = torch.ones(n) / n
        result = shannon_entropy(p)
        expected = math.log(n)
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)

    def test_delta_distribution_zero_entropy(self):
        """Delta distribution has zero entropy."""
        p = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = shannon_entropy(p)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_bernoulli_entropy(self):
        """Verify entropy for Bernoulli distribution."""
        p_val = 0.3
        p = torch.tensor([p_val, 1 - p_val])
        result = shannon_entropy(p)
        expected = -(
            p_val * math.log(p_val) + (1 - p_val) * math.log(1 - p_val)
        )
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)

    def test_matches_scipy(self):
        """Result matches scipy.stats.entropy."""
        p = torch.softmax(torch.randn(10), dim=-1)
        result = shannon_entropy(p)
        expected = scipy.stats.entropy(p.numpy())
        assert torch.isclose(
            result, torch.tensor(expected, dtype=p.dtype), rtol=1e-5
        )

    def test_matches_scipy_batch(self):
        """Result matches scipy for batched inputs."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)
        result = shannon_entropy(p)
        expected = torch.tensor(
            [scipy.stats.entropy(p[i].numpy()) for i in range(p.shape[0])],
            dtype=p.dtype,
        )
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_base_2_bits(self):
        """Entropy in bits (base 2)."""
        p = torch.ones(8) / 8
        result = shannon_entropy(p, base=2)
        expected = 3.0  # log2(8) = 3 bits
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)

    def test_base_10(self):
        """Entropy in base 10."""
        p = torch.ones(10) / 10
        result = shannon_entropy(p, base=10)
        expected = 1.0  # log10(10) = 1
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)


class TestShannonEntropyInputTypes:
    """Input type handling tests."""

    def test_log_probability_input(self):
        """Handles log_probability input type."""
        p_probs = torch.softmax(torch.randn(10), dim=-1)
        log_p = torch.log(p_probs)
        result_prob = shannon_entropy(p_probs, input_type="probability")
        result_log = shannon_entropy(log_p, input_type="log_probability")
        assert torch.isclose(result_prob, result_log, rtol=1e-5)

    def test_logits_input(self):
        """Handles logits input type."""
        logits = torch.randn(10)
        p_probs = torch.softmax(logits, dim=-1)
        result_prob = shannon_entropy(p_probs, input_type="probability")
        result_logits = shannon_entropy(logits, input_type="logits")
        assert torch.isclose(result_prob, result_logits, rtol=1e-5)


class TestShannonEntropyReduction:
    """Reduction mode tests."""

    def test_reduction_none(self):
        """reduction='none' returns per-sample entropies."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        result = shannon_entropy(p, reduction="none")
        assert result.shape == torch.Size([5])

    def test_reduction_sum(self):
        """reduction='sum' returns sum of all entropies."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        result_none = shannon_entropy(p, reduction="none")
        result_sum = shannon_entropy(p, reduction="sum")
        assert torch.isclose(result_sum, result_none.sum(), rtol=1e-5)

    def test_reduction_mean(self):
        """reduction='mean' returns mean of all entropies."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        result_none = shannon_entropy(p, reduction="none")
        result_mean = shannon_entropy(p, reduction="mean")
        assert torch.isclose(result_mean, result_none.mean(), rtol=1e-5)


class TestShannonEntropyGradients:
    """Gradient computation tests."""

    def test_gradcheck(self):
        """Gradients are correct."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return shannon_entropy(p_in, input_type="probability")

        assert gradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradgradcheck(self):
        """Second-order gradients are correct."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return shannon_entropy(p_in, input_type="probability")

        assert gradgradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass runs without errors."""
        p = torch.softmax(torch.randn(10), dim=-1)
        p.requires_grad_(True)
        result = shannon_entropy(p)
        result.backward()
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


class TestShannonEntropyEdgeCases:
    """Edge case handling tests."""

    def test_near_zero_probabilities(self):
        """Handles near-zero probabilities gracefully."""
        p = torch.tensor([0.999, 0.001])
        result = shannon_entropy(p)
        assert torch.isfinite(result)
        assert result > 0

    def test_dtype_float32(self):
        """Works with float32 inputs."""
        p = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)
        result = shannon_entropy(p)
        assert result.dtype == torch.float32
        assert torch.isfinite(result)

    def test_dtype_float64(self):
        """Works with float64 inputs."""
        p = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)
        result = shannon_entropy(p)
        assert result.dtype == torch.float64
        assert torch.isfinite(result)


class TestShannonEntropyValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Raises error for invalid input_type."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="input_type must be one of"):
            shannon_entropy(p, input_type="invalid")

    def test_invalid_reduction(self):
        """Raises error for invalid reduction."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="reduction must be one of"):
            shannon_entropy(p, reduction="invalid")

    def test_invalid_base(self):
        """Raises error for invalid base."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="base must be positive"):
            shannon_entropy(p, base=0)
        with pytest.raises(ValueError, match="base must be positive"):
            shannon_entropy(p, base=1)

    def test_non_tensor_input(self):
        """Raises error for non-tensor inputs."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            shannon_entropy([0.5, 0.5])


class TestShannonEntropyDim:
    """Dimension parameter tests."""

    def test_dim_0(self):
        """Works with dim=0."""
        p = torch.softmax(torch.randn(5, 3), dim=0)
        result = shannon_entropy(p, dim=0)
        assert result.shape == torch.Size([3])

    def test_dim_negative(self):
        """Works with negative dim values."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        result_neg = shannon_entropy(p, dim=-1)
        result_pos = shannon_entropy(p, dim=1)
        assert torch.allclose(result_neg, result_pos)


class TestShannonEntropyMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape(self):
        """Meta tensors produce correct output shapes."""
        p = torch.softmax(torch.randn(10, 5), dim=-1).to("meta")
        result = shannon_entropy(p)
        assert result.shape == torch.Size([10])
        assert result.device.type == "meta"

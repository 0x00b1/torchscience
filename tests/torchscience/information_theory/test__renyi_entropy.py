"""Comprehensive tests for Renyi entropy."""

import math

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.information import renyi_entropy, shannon_entropy


class TestRenyiEntropyBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D probability vector."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        result = renyi_entropy(p, alpha=2)
        assert result.shape == torch.Size([])

    def test_output_shape_batch(self):
        """Returns correct shape for batched input."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        result = renyi_entropy(p, alpha=2)
        assert result.shape == torch.Size([10])

    def test_output_shape_2d_dim0(self):
        """Reduction along dim=0."""
        p = torch.softmax(torch.randn(5, 10), dim=0)
        result = renyi_entropy(p, alpha=2, dim=0)
        assert result.shape == torch.Size([10])


class TestRenyiEntropyCorrectness:
    """Numerical correctness tests."""

    def test_uniform_distribution(self):
        """Uniform distribution: H_alpha = log(n) for all alpha."""
        n = 8
        p = torch.ones(n) / n

        for alpha in [0.5, 2.0, 5.0]:
            result = renyi_entropy(p, alpha=alpha)
            expected = math.log(n)
            assert torch.isclose(result, torch.tensor(expected), rtol=1e-4)

    def test_converges_to_shannon_from_below(self):
        """H_alpha -> H_1 (Shannon) as alpha -> 1 from below."""
        p = torch.softmax(torch.randn(10), dim=-1)
        h_shannon = shannon_entropy(p)

        for alpha in [0.9, 0.99, 0.999]:
            h_renyi = renyi_entropy(p, alpha=alpha)
            assert torch.isclose(h_renyi, h_shannon, rtol=0.1)

    def test_converges_to_shannon_from_above(self):
        """H_alpha -> H_1 (Shannon) as alpha -> 1 from above."""
        p = torch.softmax(torch.randn(10), dim=-1)
        h_shannon = shannon_entropy(p)

        for alpha in [1.1, 1.01, 1.001]:
            h_renyi = renyi_entropy(p, alpha=alpha)
            assert torch.isclose(h_renyi, h_shannon, rtol=0.1)

    def test_collision_entropy(self):
        """Alpha=2 gives collision entropy: -log(sum p_i^2)."""
        p = torch.tensor([0.5, 0.3, 0.2])
        result = renyi_entropy(p, alpha=2)
        expected = -math.log(0.5**2 + 0.3**2 + 0.2**2)
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)

    def test_hartley_entropy(self):
        """Alpha=0 gives Hartley entropy: log(|support|)."""
        p = torch.tensor([0.5, 0.3, 0.2, 0.0])  # Support size = 3
        result = renyi_entropy(p, alpha=0.0)
        expected = math.log(3)
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)

    def test_monotonicity_in_alpha(self):
        """H_alpha is non-increasing in alpha."""
        p = torch.softmax(torch.randn(10), dim=-1)

        h_0_5 = renyi_entropy(p, alpha=0.5)
        h_2 = renyi_entropy(p, alpha=2.0)
        h_10 = renyi_entropy(p, alpha=10.0)

        assert h_0_5 >= h_2 - 1e-5
        assert h_2 >= h_10 - 1e-5

    def test_base_conversion(self):
        """Base parameter converts entropy units correctly."""
        p = torch.tensor([0.5, 0.5])

        h_nats = renyi_entropy(p, alpha=2)
        h_bits = renyi_entropy(p, alpha=2, base=2)

        # H_bits = H_nats / log(2)
        assert torch.isclose(h_bits, h_nats / math.log(2), rtol=1e-5)


class TestRenyiEntropyGradients:
    """Gradient computation tests."""

    def test_gradcheck(self):
        """Gradients are correct for general alpha."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return renyi_entropy(p_in, alpha=2.0)

        assert gradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_alpha_0_5(self):
        """Gradients are correct for alpha=0.5."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return renyi_entropy(p_in, alpha=0.5)

        assert gradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_batched(self):
        """Gradients are correct for batched input."""
        p = torch.softmax(torch.randn(3, 5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return renyi_entropy(p_in, alpha=2.0, reduction="sum")

        assert gradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradgradcheck_alpha2(self):
        """Second-order gradients are correct for alpha=2."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return renyi_entropy(p_in, alpha=2.0, input_type="probability")

        assert gradgradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradgradcheck_batched(self):
        """Second-order gradients work with batched inputs."""
        p = torch.softmax(torch.randn(3, 5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return renyi_entropy(
                p_in, alpha=2.0, input_type="probability", reduction="sum"
            )

        assert gradgradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("alpha", [0.5, 1.5, 2.0, 3.0, 5.0])
    def test_gradgradcheck_various_alpha(self, alpha):
        """Second-order gradients work for various alpha values."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return renyi_entropy(p_in, alpha=alpha, input_type="probability")

        assert gradgradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)


class TestRenyiEntropyInputTypes:
    """Input type handling tests."""

    def test_logits_input(self):
        """Correctly handles logits input."""
        logits = torch.randn(10)
        p = torch.softmax(logits, dim=-1)

        h_prob = renyi_entropy(p, alpha=2, input_type="probability")
        h_logits = renyi_entropy(logits, alpha=2, input_type="logits")

        assert torch.isclose(h_prob, h_logits, rtol=1e-5)

    def test_log_probability_input(self):
        """Correctly handles log probability input."""
        p = torch.softmax(torch.randn(10), dim=-1)
        log_p = torch.log(p)

        h_prob = renyi_entropy(p, alpha=2, input_type="probability")
        h_log = renyi_entropy(log_p, alpha=2, input_type="log_probability")

        assert torch.isclose(h_prob, h_log, rtol=1e-5)


class TestRenyiEntropyReductions:
    """Reduction mode tests."""

    def test_reduction_none(self):
        """None reduction returns per-sample values."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        result = renyi_entropy(p, alpha=2, reduction="none")
        assert result.shape == torch.Size([10])

    def test_reduction_mean(self):
        """Mean reduction returns scalar."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        result_none = renyi_entropy(p, alpha=2, reduction="none")
        result_mean = renyi_entropy(p, alpha=2, reduction="mean")

        assert result_mean.shape == torch.Size([])
        assert torch.isclose(result_mean, result_none.mean(), rtol=1e-5)

    def test_reduction_sum(self):
        """Sum reduction returns scalar."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        result_none = renyi_entropy(p, alpha=2, reduction="none")
        result_sum = renyi_entropy(p, alpha=2, reduction="sum")

        assert result_sum.shape == torch.Size([])
        assert torch.isclose(result_sum, result_none.sum(), rtol=1e-5)


class TestRenyiEntropyValidation:
    """Input validation tests."""

    def test_invalid_alpha_negative(self):
        """Raises error for negative alpha."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="alpha must be >= 0"):
            renyi_entropy(p, alpha=-1)

    def test_invalid_alpha_one(self):
        """Raises error for alpha=1."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="alpha cannot be 1"):
            renyi_entropy(p, alpha=1.0)

    def test_invalid_input_type(self):
        """Raises error for invalid input_type."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="input_type must be one of"):
            renyi_entropy(p, alpha=2, input_type="invalid")

    def test_invalid_reduction(self):
        """Raises error for invalid reduction."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="reduction must be one of"):
            renyi_entropy(p, alpha=2, reduction="invalid")

    def test_invalid_base(self):
        """Raises error for invalid base."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="base must be positive"):
            renyi_entropy(p, alpha=2, base=-1)
        with pytest.raises(ValueError, match="base must be positive"):
            renyi_entropy(p, alpha=2, base=1)

    def test_non_tensor_input(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="p must be a Tensor"):
            renyi_entropy([0.5, 0.5], alpha=2)

    def test_invalid_dim(self):
        """Raises error for out of range dim."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(IndexError, match="dim .* out of range"):
            renyi_entropy(p, alpha=2, dim=2)


class TestRenyiEntropyMeta:
    """Meta tensor support tests."""

    def test_meta_tensor_shape(self):
        """Meta tensors return correct shape."""
        p = torch.softmax(torch.randn(10, 5, device="meta"), dim=-1)
        result = renyi_entropy(p, alpha=2)
        assert result.shape == torch.Size([10])
        assert result.device.type == "meta"


class TestRenyiEntropyDtypes:
    """Data type support tests."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Supports various floating point dtypes."""
        p = torch.softmax(torch.randn(10, dtype=dtype), dim=-1)
        result = renyi_entropy(p, alpha=2)
        assert result.dtype == dtype

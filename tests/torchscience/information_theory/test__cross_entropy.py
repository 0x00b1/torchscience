"""Comprehensive tests for cross-entropy."""

import pytest
import scipy.stats
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.information_theory import (
    cross_entropy,
    kullback_leibler_divergence,
    shannon_entropy,
)


class TestCrossEntropyBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D probability vectors."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.5, 0.25, 0.125, 0.125])
        result = cross_entropy(p, q)
        assert result.shape == torch.Size([])

    def test_output_shape_2d_batch(self):
        """Returns 1D tensor for batch of distributions."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)
        result = cross_entropy(p, q)
        assert result.shape == torch.Size([10])

    def test_output_shape_3d_batch(self):
        """Returns 2D tensor for nested batch of distributions."""
        p = torch.softmax(torch.randn(4, 5, 8), dim=-1)
        q = torch.softmax(torch.randn(4, 5, 8), dim=-1)
        result = cross_entropy(p, q)
        assert result.shape == torch.Size([4, 5])

    def test_non_negativity(self):
        """Cross-entropy is always non-negative for valid distributions."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(100), dim=-1)
            q = torch.softmax(torch.randn(100), dim=-1)
            result = cross_entropy(p, q)
            assert result >= 0, (
                f"Cross-entropy should be non-negative, got {result}"
            )


class TestCrossEntropyCorrectness:
    """Numerical correctness tests."""

    def test_self_cross_entropy_equals_entropy(self):
        """Cross-entropy H(P, P) equals Shannon entropy H(P)."""
        p = torch.softmax(torch.randn(10), dim=-1)
        ce_pp = cross_entropy(p, p)
        h_p = shannon_entropy(p)
        assert torch.isclose(ce_pp, h_p, rtol=1e-5)

    def test_decomposition_entropy_plus_kl(self):
        """Verify H(P, Q) = H(P) + D_KL(P || Q)."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        ce = cross_entropy(p, q)
        h = shannon_entropy(p)
        kl = kullback_leibler_divergence(p, q)

        assert torch.isclose(ce, h + kl, rtol=1e-5)

    def test_decomposition_batch(self):
        """Decomposition holds for batched inputs."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)
        q = torch.softmax(torch.randn(5, 8), dim=-1)

        ce = cross_entropy(p, q)
        h = shannon_entropy(p)
        kl = kullback_leibler_divergence(p, q)

        assert torch.allclose(ce, h + kl, rtol=1e-5)

    def test_matches_scipy(self):
        """Result matches scipy-based calculation: H(P) + D_KL(P||Q)."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)
        result = cross_entropy(p, q)
        # scipy.stats.entropy(p) = H(P), scipy.stats.entropy(p, q) = D_KL(P||Q)
        # Cross-entropy H(P, Q) = H(P) + D_KL(P||Q)
        entropy_p = scipy.stats.entropy(p.numpy())
        kl_div = scipy.stats.entropy(p.numpy(), q.numpy())
        expected = entropy_p + kl_div
        assert torch.isclose(
            result, torch.tensor(expected, dtype=p.dtype), rtol=1e-5
        )

    def test_matches_scipy_batch(self):
        """Result matches scipy-based calculation for batched inputs."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)
        q = torch.softmax(torch.randn(5, 8), dim=-1)
        result = cross_entropy(p, q)
        expected = torch.tensor(
            [
                scipy.stats.entropy(p[i].numpy())
                + scipy.stats.entropy(p[i].numpy(), q[i].numpy())
                for i in range(p.shape[0])
            ],
            dtype=p.dtype,
        )
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_base_2_bits(self):
        """Cross-entropy in bits (base 2)."""
        p = torch.ones(8) / 8
        q = torch.ones(8) / 8
        result = cross_entropy(p, q, base=2)
        expected = 3.0  # log2(8) = 3 bits (self cross-entropy = entropy)
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)

    def test_asymmetry(self):
        """Cross-entropy is asymmetric: H(P,Q) != H(Q,P)."""
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.2, 0.3, 0.5])

        ce_pq = cross_entropy(p, q)
        ce_qp = cross_entropy(q, p)

        assert not torch.isclose(ce_pq, ce_qp, rtol=1e-3)


class TestCrossEntropyInputTypes:
    """Input type handling tests."""

    def test_log_probability_input(self):
        """Handles log_probability input type."""
        p_probs = torch.softmax(torch.randn(10), dim=-1)
        q_probs = torch.softmax(torch.randn(10), dim=-1)
        log_p = torch.log(p_probs)
        log_q = torch.log(q_probs)
        result_prob = cross_entropy(p_probs, q_probs, input_type="probability")
        result_log = cross_entropy(log_p, log_q, input_type="log_probability")
        assert torch.isclose(result_prob, result_log, rtol=1e-5)

    def test_logits_input(self):
        """Handles logits input type."""
        logits_p = torch.randn(10)
        logits_q = torch.randn(10)
        p_probs = torch.softmax(logits_p, dim=-1)
        q_probs = torch.softmax(logits_q, dim=-1)
        result_prob = cross_entropy(p_probs, q_probs, input_type="probability")
        result_logits = cross_entropy(logits_p, logits_q, input_type="logits")
        assert torch.isclose(result_prob, result_logits, rtol=1e-5)


class TestCrossEntropyReduction:
    """Reduction mode tests."""

    def test_reduction_none(self):
        """reduction='none' returns per-sample cross-entropies."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)
        result = cross_entropy(p, q, reduction="none")
        assert result.shape == torch.Size([5])

    def test_reduction_sum(self):
        """reduction='sum' returns sum of all cross-entropies."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)
        result_none = cross_entropy(p, q, reduction="none")
        result_sum = cross_entropy(p, q, reduction="sum")
        assert torch.isclose(result_sum, result_none.sum(), rtol=1e-5)

    def test_reduction_mean(self):
        """reduction='mean' returns mean of all cross-entropies."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)
        result_none = cross_entropy(p, q, reduction="none")
        result_mean = cross_entropy(p, q, reduction="mean")
        assert torch.isclose(result_mean, result_none.mean(), rtol=1e-5)


class TestCrossEntropyGradients:
    """Gradient computation tests."""

    def test_gradcheck_p(self):
        """Gradients w.r.t. p are correct."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)

        def func(p_in):
            return cross_entropy(p_in, q, input_type="probability")

        assert gradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_q(self):
        """Gradients w.r.t. q are correct."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q.requires_grad_(True)

        def func(q_in):
            return cross_entropy(p, q_in, input_type="probability")

        assert gradcheck(func, (q,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradgradcheck(self):
        """Second-order gradients are correct."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)
        q.requires_grad_(True)

        def func(p_in, q_in):
            return cross_entropy(p_in, q_in, input_type="probability")

        assert gradgradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass runs without errors."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)
        p.requires_grad_(True)
        q.requires_grad_(True)
        result = cross_entropy(p, q)
        result.backward()
        assert p.grad is not None
        assert q.grad is not None
        assert torch.isfinite(p.grad).all()
        assert torch.isfinite(q.grad).all()


class TestCrossEntropyEdgeCases:
    """Edge case handling tests."""

    def test_near_zero_q_probabilities(self):
        """Handles near-zero q probabilities gracefully."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.999, 0.001])
        result = cross_entropy(p, q)
        assert torch.isfinite(result)
        assert result > 0

    def test_dtype_float32(self):
        """Works with float32 inputs."""
        p = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)
        result = cross_entropy(p, q)
        assert result.dtype == torch.float32
        assert torch.isfinite(result)

    def test_dtype_float64(self):
        """Works with float64 inputs."""
        p = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)
        result = cross_entropy(p, q)
        assert result.dtype == torch.float64
        assert torch.isfinite(result)


class TestCrossEntropyValidation:
    """Input validation tests."""

    def test_shape_mismatch(self):
        """Raises error for shape mismatch between p and q."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.3, 0.3, 0.4])
        with pytest.raises(ValueError, match="same shape"):
            cross_entropy(p, q)

    def test_invalid_input_type(self):
        """Raises error for invalid input_type."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="input_type must be one of"):
            cross_entropy(p, q, input_type="invalid")

    def test_invalid_reduction(self):
        """Raises error for invalid reduction."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="reduction must be one of"):
            cross_entropy(p, q, reduction="invalid")

    def test_invalid_base(self):
        """Raises error for invalid base."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="base must be positive"):
            cross_entropy(p, q, base=0)
        with pytest.raises(ValueError, match="base must be positive"):
            cross_entropy(p, q, base=1)

    def test_non_tensor_p(self):
        """Raises error for non-tensor p input."""
        q = torch.tensor([0.5, 0.5])
        with pytest.raises(TypeError, match="p must be a Tensor"):
            cross_entropy([0.5, 0.5], q)

    def test_non_tensor_q(self):
        """Raises error for non-tensor q input."""
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(TypeError, match="q must be a Tensor"):
            cross_entropy(p, [0.5, 0.5])


class TestCrossEntropyDim:
    """Dimension parameter tests."""

    def test_dim_0(self):
        """Works with dim=0."""
        p = torch.softmax(torch.randn(5, 3), dim=0)
        q = torch.softmax(torch.randn(5, 3), dim=0)
        result = cross_entropy(p, q, dim=0)
        assert result.shape == torch.Size([3])

    def test_dim_negative(self):
        """Works with negative dim values."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(3, 5), dim=-1)
        result_neg = cross_entropy(p, q, dim=-1)
        result_pos = cross_entropy(p, q, dim=1)
        assert torch.allclose(result_neg, result_pos)


class TestCrossEntropyMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape(self):
        """Meta tensors produce correct output shapes."""
        p = torch.softmax(torch.randn(10, 5), dim=-1).to("meta")
        q = torch.softmax(torch.randn(10, 5), dim=-1).to("meta")
        result = cross_entropy(p, q)
        assert result.shape == torch.Size([10])
        assert result.device.type == "meta"

"""Comprehensive tests for Renyi divergence."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information_theory import (
    kullback_leibler_divergence,
    renyi_divergence,
)


class TestRenyiDivergenceBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D probability vectors."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.1, 0.2, 0.3, 0.4])

        result = renyi_divergence(p, q, alpha=2.0)

        assert result.shape == torch.Size([])

    def test_output_shape_2d_batch(self):
        """Returns 1D tensor for batch of distributions."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        result = renyi_divergence(p, q, alpha=2.0)

        assert result.shape == torch.Size([10])

    def test_output_shape_3d_batch(self):
        """Returns 2D tensor for nested batch of distributions."""
        p = torch.softmax(torch.randn(4, 5, 8), dim=-1)
        q = torch.softmax(torch.randn(4, 5, 8), dim=-1)

        result = renyi_divergence(p, q, alpha=2.0)

        assert result.shape == torch.Size([4, 5])

    def test_non_negativity(self):
        """Renyi divergence is always non-negative for alpha > 0."""
        torch.manual_seed(42)
        for alpha in [0.5, 2.0, 5.0]:
            for _ in range(5):
                p = torch.softmax(torch.randn(100), dim=-1)
                q = torch.softmax(torch.randn(100), dim=-1)

                result = renyi_divergence(p, q, alpha=alpha)

                assert result >= -1e-6, (
                    f"Renyi divergence with alpha={alpha} should be non-negative, "
                    f"got {result}"
                )

    def test_zero_for_identical_distributions(self):
        """Renyi divergence is zero when P equals Q."""
        p = torch.tensor([0.2, 0.3, 0.5])
        q = p.clone()

        result = renyi_divergence(p, q, alpha=2.0)

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)


class TestRenyiDivergenceCorrectness:
    """Numerical correctness tests."""

    def test_manual_calculation_alpha2(self):
        """Verify against manual calculation for alpha=2."""
        p = torch.tensor([0.5, 0.3, 0.2])
        q = torch.tensor([0.4, 0.4, 0.2])

        # D_2(P||Q) = log(sum p_i^2 / q_i)
        # = log(0.25/0.4 + 0.09/0.4 + 0.04/0.2)
        # = log(0.625 + 0.225 + 0.2)
        # = log(1.05)
        sum_term = (0.5**2 / 0.4) + (0.3**2 / 0.4) + (0.2**2 / 0.2)
        expected = math.log(sum_term)

        result = renyi_divergence(p, q, alpha=2.0)

        assert torch.isclose(result, torch.tensor(expected), rtol=1e-4)

    def test_manual_calculation_alpha_half(self):
        """Verify against manual calculation for alpha=0.5."""
        p = torch.tensor([0.6, 0.4])
        q = torch.tensor([0.3, 0.7])

        # D_0.5(P||Q) = -2 * log(sum sqrt(p_i * q_i))
        # = -2 * log(sqrt(0.6*0.3) + sqrt(0.4*0.7))
        sum_term = math.sqrt(0.6 * 0.3) + math.sqrt(0.4 * 0.7)
        expected = -2 * math.log(sum_term)

        result = renyi_divergence(p, q, alpha=0.5)

        assert torch.isclose(result, torch.tensor(expected), rtol=1e-4)

    def test_asymmetric(self):
        """Renyi divergence is asymmetric: D_alpha(P||Q) != D_alpha(Q||P)."""
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.2, 0.3, 0.5])

        d_pq = renyi_divergence(p, q, alpha=2.0)
        d_qp = renyi_divergence(q, p, alpha=2.0)

        assert not torch.isclose(d_pq, d_qp, rtol=1e-3)

    def test_increases_with_alpha(self):
        """For non-identical distributions, D_alpha generally changes with alpha."""
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.2, 0.3, 0.5])

        d_2 = renyi_divergence(p, q, alpha=2.0)
        d_5 = renyi_divergence(p, q, alpha=5.0)

        # They should be different
        assert not torch.isclose(d_2, d_5, rtol=0.01)


class TestRenyiDivergenceSpecialCases:
    """Tests for special alpha values and their relationships."""

    def test_alpha_near_kl_divergence(self):
        """Alpha close to 1 approximates KL divergence."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        kl = kullback_leibler_divergence(p, q)

        # Test increasingly close alpha values
        for alpha in [1.1, 1.01, 1.001]:
            renyi = renyi_divergence(p, q, alpha=alpha)
            # The closer alpha is to 1, the closer to KL
            assert torch.isclose(renyi, kl, rtol=0.5), (
                f"alpha={alpha}: Renyi={renyi.item():.4f}, KL={kl.item():.4f}"
            )

    def test_alpha_0_5_bhattacharyya_relation(self):
        """Alpha=0.5 relates to Bhattacharyya coefficient.

        D_0.5(P||Q) = -2 * log(BC(P,Q))
        where BC(P,Q) = sum sqrt(p_i * q_i) is the Bhattacharyya coefficient.
        """
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        d_half = renyi_divergence(p, q, alpha=0.5)

        # Compute Bhattacharyya coefficient
        bc = torch.sqrt(p * q).sum()
        expected = -2 * torch.log(bc)

        assert torch.isclose(d_half, expected, rtol=1e-4)


class TestRenyiDivergenceInputTypes:
    """Input type handling tests."""

    def test_log_probability_input(self):
        """Handles log_probability input type."""
        p_probs = torch.softmax(torch.randn(10), dim=-1)
        q_probs = torch.softmax(torch.randn(10), dim=-1)

        log_p = torch.log(p_probs)
        log_q = torch.log(q_probs)

        result_prob = renyi_divergence(
            p_probs, q_probs, alpha=2.0, input_type="probability"
        )
        result_log = renyi_divergence(
            log_p, log_q, alpha=2.0, input_type="log_probability"
        )

        assert torch.isclose(result_prob, result_log, rtol=1e-4)

    def test_logits_input(self):
        """Handles logits input type (applies softmax internally)."""
        logits_p = torch.randn(10)
        logits_q = torch.randn(10)

        p_probs = torch.softmax(logits_p, dim=-1)
        q_probs = torch.softmax(logits_q, dim=-1)

        result_prob = renyi_divergence(
            p_probs, q_probs, alpha=2.0, input_type="probability"
        )
        result_logits = renyi_divergence(
            logits_p, logits_q, alpha=2.0, input_type="logits"
        )

        assert torch.isclose(result_prob, result_logits, rtol=1e-4)


class TestRenyiDivergenceReduction:
    """Reduction mode tests."""

    def test_reduction_none(self):
        """reduction='none' returns per-sample divergences."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result = renyi_divergence(p, q, alpha=2.0, reduction="none")

        assert result.shape == torch.Size([5])

    def test_reduction_sum(self):
        """reduction='sum' returns sum of all divergences."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result_none = renyi_divergence(p, q, alpha=2.0, reduction="none")
        result_sum = renyi_divergence(p, q, alpha=2.0, reduction="sum")

        assert torch.isclose(result_sum, result_none.sum(), rtol=1e-5)

    def test_reduction_mean(self):
        """reduction='mean' returns mean over all elements."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result_none = renyi_divergence(p, q, alpha=2.0, reduction="none")
        result_mean = renyi_divergence(p, q, alpha=2.0, reduction="mean")

        assert torch.isclose(result_mean, result_none.mean(), rtol=1e-5)

    def test_reduction_batchmean(self):
        """reduction='batchmean' returns mean over batch dimension."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result_none = renyi_divergence(p, q, alpha=2.0, reduction="none")
        result_batchmean = renyi_divergence(
            p, q, alpha=2.0, reduction="batchmean"
        )

        expected = result_none.sum() / p.shape[0]
        assert torch.isclose(result_batchmean, expected, rtol=1e-5)


class TestRenyiDivergenceBase:
    """Logarithm base tests."""

    def test_base_2_bits(self):
        """Base 2 gives result in bits."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.25, 0.75])

        result_nats = renyi_divergence(p, q, alpha=2.0, base=None)
        result_bits = renyi_divergence(p, q, alpha=2.0, base=2.0)

        # result_bits = result_nats / log(2)
        expected = result_nats / math.log(2)

        assert torch.isclose(result_bits, expected, rtol=1e-5)

    def test_base_10_hartleys(self):
        """Base 10 gives result in hartleys (bans)."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.25, 0.75])

        result_nats = renyi_divergence(p, q, alpha=2.0, base=None)
        result_hartleys = renyi_divergence(p, q, alpha=2.0, base=10.0)

        expected = result_nats / math.log(10)

        assert torch.isclose(result_hartleys, expected, rtol=1e-5)


class TestRenyiDivergencePairwise:
    """Pairwise divergence computation tests."""

    def test_pairwise_output_shape(self):
        """pairwise=True returns (m, k) matrix for (m, n) and (k, n) inputs."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(4, 5), dim=-1)

        result = renyi_divergence(p, q, alpha=2.0, pairwise=True)

        assert result.shape == torch.Size([3, 4])

    def test_pairwise_values_correctness(self):
        """Pairwise values match individual computations."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(4, 5), dim=-1)

        result = renyi_divergence(p, q, alpha=2.0, pairwise=True)

        for i in range(3):
            for j in range(4):
                expected = renyi_divergence(p[i], q[j], alpha=2.0)
                assert torch.isclose(result[i, j], expected, rtol=1e-4), (
                    f"Mismatch at [{i}, {j}]: got {result[i, j]}, "
                    f"expected {expected}"
                )

    def test_pairwise_diagonal_self_divergence(self):
        """Pairwise diagonal is zero when p == q."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)

        result = renyi_divergence(p, p, alpha=2.0, pairwise=True)

        diagonal = torch.diag(result)
        assert torch.allclose(diagonal, torch.zeros(5), atol=1e-5)


class TestRenyiDivergenceGradients:
    """Gradient computation tests."""

    def test_gradcheck_probability(self):
        """Gradients are correct for probability inputs."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        def func(p_in, q_in):
            return renyi_divergence(
                p_in, q_in, alpha=2.0, input_type="probability"
            )

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_alpha_half(self):
        """Gradients are correct for alpha=0.5."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        def func(p_in, q_in):
            return renyi_divergence(
                p_in, q_in, alpha=0.5, input_type="probability"
            )

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_batch(self):
        """Gradients are correct for batched inputs."""
        p = torch.softmax(torch.randn(3, 5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(3, 5, dtype=torch.float64), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        def func(p_in, q_in):
            return renyi_divergence(
                p_in,
                q_in,
                alpha=2.0,
                input_type="probability",
                reduction="sum",
            )

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass runs without errors."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        result = renyi_divergence(p, q, alpha=2.0)
        result.backward()

        assert p.grad is not None
        assert q.grad is not None
        assert torch.isfinite(p.grad).all()
        assert torch.isfinite(q.grad).all()


class TestRenyiDivergenceValidation:
    """Input validation tests."""

    def test_invalid_alpha_one(self):
        """Raises error for alpha=1."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="alpha cannot be 1"):
            renyi_divergence(p, q, alpha=1.0)

    def test_invalid_alpha_negative(self):
        """Raises error for negative alpha."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="alpha must be >= 0"):
            renyi_divergence(p, q, alpha=-1.0)

    def test_invalid_input_type(self):
        """Raises error for invalid input_type."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="input_type must be one of"):
            renyi_divergence(p, q, alpha=2.0, input_type="invalid")

    def test_invalid_reduction(self):
        """Raises error for invalid reduction."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="reduction must be one of"):
            renyi_divergence(p, q, alpha=2.0, reduction="invalid")

    def test_invalid_base(self):
        """Raises error for invalid base values."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="base must be positive"):
            renyi_divergence(p, q, alpha=2.0, base=0.0)

        with pytest.raises(ValueError, match="base must be positive"):
            renyi_divergence(p, q, alpha=2.0, base=-1.0)

        with pytest.raises(ValueError, match="base must be positive"):
            renyi_divergence(p, q, alpha=2.0, base=1.0)

    def test_mismatched_sizes(self):
        """Raises error for mismatched distribution sizes."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.3, 0.3, 0.4])

        with pytest.raises(ValueError, match="Distribution sizes must match"):
            renyi_divergence(p, q, alpha=2.0)

    def test_invalid_dim(self):
        """Raises error for invalid dim."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(IndexError, match="dim .* out of range"):
            renyi_divergence(p, q, alpha=2.0, dim=5)

    def test_non_tensor_input(self):
        """Raises error for non-tensor inputs."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            renyi_divergence([0.5, 0.5], torch.tensor([0.5, 0.5]), alpha=2.0)

    def test_pairwise_requires_2d(self):
        """pairwise=True requires at least 2D inputs."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="pairwise=True requires"):
            renyi_divergence(p, q, alpha=2.0, pairwise=True)


class TestRenyiDivergenceMeta:
    """Meta tensor support tests."""

    def test_meta_tensor_shape(self):
        """Meta tensors return correct shape."""
        p = torch.softmax(torch.randn(10, 5, device="meta"), dim=-1)
        q = torch.softmax(torch.randn(10, 5, device="meta"), dim=-1)

        result = renyi_divergence(p, q, alpha=2.0)

        assert result.shape == torch.Size([10])
        assert result.device.type == "meta"

    def test_meta_tensor_pairwise_shape(self):
        """Meta tensors return correct pairwise shape."""
        p = torch.softmax(torch.randn(3, 5, device="meta"), dim=-1)
        q = torch.softmax(torch.randn(4, 5, device="meta"), dim=-1)

        result = renyi_divergence(p, q, alpha=2.0, pairwise=True)

        assert result.shape == torch.Size([3, 4])
        assert result.device.type == "meta"


class TestRenyiDivergenceDtypes:
    """Data type support tests."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Supports various floating point dtypes."""
        p = torch.softmax(torch.randn(10, dtype=dtype), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=dtype), dim=-1)

        result = renyi_divergence(p, q, alpha=2.0)

        assert result.dtype == dtype
        assert torch.isfinite(result)

    def test_dtype_promotion(self):
        """Promotes dtypes when inputs differ."""
        p = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)

        result = renyi_divergence(p, q, alpha=2.0)

        assert result.dtype == torch.float64


class TestRenyiDivergenceDim:
    """Dimension parameter tests."""

    def test_dim_0(self):
        """Works with dim=0 for distribution along first axis."""
        p = torch.softmax(torch.randn(5, 3), dim=0)
        q = torch.softmax(torch.randn(5, 3), dim=0)

        result = renyi_divergence(p, q, alpha=2.0, dim=0)

        assert result.shape == torch.Size([3])

    def test_dim_negative(self):
        """Works with negative dim values."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(3, 5), dim=-1)

        result_neg = renyi_divergence(p, q, alpha=2.0, dim=-1)
        result_pos = renyi_divergence(p, q, alpha=2.0, dim=1)

        assert torch.allclose(result_neg, result_pos)

    def test_dim_middle(self):
        """Works with middle dimension."""
        p = torch.softmax(torch.randn(2, 5, 3), dim=1)
        q = torch.softmax(torch.randn(2, 5, 3), dim=1)

        result = renyi_divergence(p, q, alpha=2.0, dim=1)

        assert result.shape == torch.Size([2, 3])

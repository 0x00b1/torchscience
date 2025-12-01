"""Comprehensive tests for Kullback-Leibler divergence."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from torchscience.information_theory import kullback_leibler_divergence


class TestKLDivergenceBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D probability vectors."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.1, 0.2, 0.3, 0.4])

        result = kullback_leibler_divergence(p, q)

        assert result.shape == torch.Size([])

    def test_output_shape_2d_batch(self):
        """Returns 1D tensor for batch of distributions."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        result = kullback_leibler_divergence(p, q)

        assert result.shape == torch.Size([10])

    def test_output_shape_3d_batch(self):
        """Returns 2D tensor for nested batch of distributions."""
        p = torch.softmax(torch.randn(4, 5, 8), dim=-1)
        q = torch.softmax(torch.randn(4, 5, 8), dim=-1)

        result = kullback_leibler_divergence(p, q)

        assert result.shape == torch.Size([4, 5])

    def test_non_negativity(self):
        """KL divergence is always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(100), dim=-1)
            q = torch.softmax(torch.randn(100), dim=-1)

            result = kullback_leibler_divergence(p, q)

            assert result >= 0, (
                f"KL divergence should be non-negative, got {result}"
            )

    def test_zero_for_identical_distributions(self):
        """KL divergence is zero when P equals Q."""
        p = torch.tensor([0.2, 0.3, 0.5])
        q = p.clone()

        result = kullback_leibler_divergence(p, q)

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-7)

    def test_zero_for_identical_batch(self):
        """KL divergence is zero for each identical pair in batch."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = p.clone()

        result = kullback_leibler_divergence(p, q)

        assert torch.allclose(result, torch.zeros(5), atol=1e-6)


class TestKLDivergenceCorrectness:
    """Numerical correctness tests."""

    def test_uniform_distributions(self):
        """Uniform vs uniform gives zero."""
        p = torch.ones(4) / 4
        q = torch.ones(4) / 4

        result = kullback_leibler_divergence(p, q)

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-7)

    def test_known_bernoulli_value(self):
        """Verify KL divergence for known Bernoulli distributions.

        For P = Bernoulli(0.5) and Q = Bernoulli(0.1):
        D_KL(P || Q) = 0.5 * log(0.5/0.1) + 0.5 * log(0.5/0.9)
                     = 0.5 * log(5) + 0.5 * log(5/9)
                     = 0.5 * (log(5) + log(5) - log(9))
                     = 0.5 * (2*log(5) - log(9))
        """
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.1, 0.9])

        result = kullback_leibler_divergence(p, q)

        expected = 0.5 * math.log(0.5 / 0.1) + 0.5 * math.log(0.5 / 0.9)
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)

    def test_matches_torch_kl_div(self):
        """Result matches torch.nn.functional.kl_div for log inputs."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        # Our implementation with probability inputs
        result = kullback_leibler_divergence(p, q)

        # PyTorch's kl_div expects log(q) as input, p as target
        # and computes sum(p * (log(p) - input)) with reduction='sum'
        log_q = torch.log(q)
        torch_result = F.kl_div(log_q, p, reduction="sum")

        assert torch.isclose(result, torch_result, rtol=1e-5)

    def test_asymmetric(self):
        """KL divergence is asymmetric: D_KL(P||Q) != D_KL(Q||P)."""
        # Use truly asymmetric distributions (not mirror images)
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.2, 0.3, 0.5])

        kl_pq = kullback_leibler_divergence(p, q)
        kl_qp = kullback_leibler_divergence(q, p)

        assert not torch.isclose(kl_pq, kl_qp, rtol=1e-3)

    def test_extreme_distributions(self):
        """Handle near-deterministic distributions correctly."""
        p = torch.tensor([0.99, 0.01])
        q = torch.tensor([0.01, 0.99])

        result = kullback_leibler_divergence(p, q)

        # Both terms should contribute significantly
        expected = 0.99 * math.log(0.99 / 0.01) + 0.01 * math.log(0.01 / 0.99)
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-3)


class TestKLDivergenceInputTypes:
    """Input type handling tests."""

    def test_log_probability_input(self):
        """Handles log_probability input type."""
        p_probs = torch.softmax(torch.randn(10), dim=-1)
        q_probs = torch.softmax(torch.randn(10), dim=-1)

        log_p = torch.log(p_probs)
        log_q = torch.log(q_probs)

        result_prob = kullback_leibler_divergence(
            p_probs, q_probs, input_type="probability"
        )
        result_log = kullback_leibler_divergence(
            log_p, log_q, input_type="log_probability"
        )

        assert torch.isclose(result_prob, result_log, rtol=1e-5)

    def test_logits_input(self):
        """Handles logits input type (applies softmax internally)."""
        logits_p = torch.randn(10)
        logits_q = torch.randn(10)

        p_probs = torch.softmax(logits_p, dim=-1)
        q_probs = torch.softmax(logits_q, dim=-1)

        result_prob = kullback_leibler_divergence(
            p_probs, q_probs, input_type="probability"
        )
        result_logits = kullback_leibler_divergence(
            logits_p, logits_q, input_type="logits"
        )

        assert torch.isclose(result_prob, result_logits, rtol=1e-5)

    def test_logits_batch(self):
        """Logits input works with batched data."""
        logits_p = torch.randn(5, 8)
        logits_q = torch.randn(5, 8)

        p_probs = torch.softmax(logits_p, dim=-1)
        q_probs = torch.softmax(logits_q, dim=-1)

        result_prob = kullback_leibler_divergence(
            p_probs, q_probs, input_type="probability"
        )
        result_logits = kullback_leibler_divergence(
            logits_p, logits_q, input_type="logits"
        )

        assert torch.allclose(result_prob, result_logits, rtol=1e-5)


class TestKLDivergenceReduction:
    """Reduction mode tests."""

    def test_reduction_none(self):
        """reduction='none' returns per-sample divergences."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result = kullback_leibler_divergence(p, q, reduction="none")

        assert result.shape == torch.Size([5])

    def test_reduction_sum(self):
        """reduction='sum' returns sum of all divergences."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result_none = kullback_leibler_divergence(p, q, reduction="none")
        result_sum = kullback_leibler_divergence(p, q, reduction="sum")

        assert torch.isclose(result_sum, result_none.sum(), rtol=1e-5)

    def test_reduction_mean(self):
        """reduction='mean' returns mean over all elements."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result_none = kullback_leibler_divergence(p, q, reduction="none")
        result_mean = kullback_leibler_divergence(p, q, reduction="mean")

        assert torch.isclose(result_mean, result_none.mean(), rtol=1e-5)

    def test_reduction_batchmean(self):
        """reduction='batchmean' returns mean over batch dimension."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result_none = kullback_leibler_divergence(p, q, reduction="none")
        result_batchmean = kullback_leibler_divergence(
            p, q, reduction="batchmean"
        )

        # batchmean divides by the number of samples in the batch
        expected = result_none.sum() / p.shape[0]
        assert torch.isclose(result_batchmean, expected, rtol=1e-5)


class TestKLDivergencePairwise:
    """Pairwise divergence computation tests."""

    def test_pairwise_output_shape(self):
        """pairwise=True returns (m, k) matrix for (m, n) and (k, n) inputs."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(4, 5), dim=-1)

        result = kullback_leibler_divergence(p, q, pairwise=True)

        assert result.shape == torch.Size([3, 4])

    def test_pairwise_values_correctness(self):
        """Pairwise values match individual computations."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(4, 5), dim=-1)

        result = kullback_leibler_divergence(p, q, pairwise=True)

        # Compare with loop-based computation
        for i in range(3):
            for j in range(4):
                expected = kullback_leibler_divergence(p[i], q[j])
                assert torch.isclose(result[i, j], expected, rtol=1e-5), (
                    f"Mismatch at [{i}, {j}]: got {result[i, j]}, "
                    f"expected {expected}"
                )

    def test_pairwise_diagonal_self_divergence(self):
        """Pairwise diagonal is zero when p == q."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)

        result = kullback_leibler_divergence(p, p, pairwise=True)

        # Diagonal should be close to zero
        diagonal = torch.diag(result)
        assert torch.allclose(diagonal, torch.zeros(5), atol=1e-6)

    @pytest.mark.skip(
        reason="Pairwise mode currently only supports 2D tensors"
    )
    def test_pairwise_batch_3d(self):
        """Pairwise works with 3D batched input."""
        p = torch.softmax(torch.randn(2, 3, 5), dim=-1)
        q = torch.softmax(torch.randn(2, 4, 5), dim=-1)

        result = kullback_leibler_divergence(p, q, pairwise=True)

        assert result.shape == torch.Size([2, 3, 4])


class TestKLDivergenceGradients:
    """Gradient computation tests."""

    def test_gradcheck_probability(self):
        """Gradients are correct for probability inputs."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        def func(p_in, q_in):
            return kullback_leibler_divergence(
                p_in, q_in, input_type="probability"
            )

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    @pytest.mark.skip(
        reason="Gradients through softmax not tracked in logits mode"
    )
    def test_gradcheck_logits(self):
        """Gradients are correct for logits inputs."""
        p = torch.randn(5, dtype=torch.float64, requires_grad=True)
        q = torch.randn(5, dtype=torch.float64, requires_grad=True)

        def func(p_in, q_in):
            return kullback_leibler_divergence(p_in, q_in, input_type="logits")

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_batch(self):
        """Gradients are correct for batched inputs."""
        p = torch.softmax(torch.randn(3, 5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(3, 5, dtype=torch.float64), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        def func(p_in, q_in):
            return kullback_leibler_divergence(
                p_in, q_in, input_type="probability", reduction="sum"
            )

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass runs without errors."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        result = kullback_leibler_divergence(p, q)
        result.backward()

        assert p.grad is not None
        assert q.grad is not None
        assert torch.isfinite(p.grad).all()
        assert torch.isfinite(q.grad).all()


class TestKLDivergenceEdgeCases:
    """Edge case handling tests."""

    def test_near_zero_probabilities(self):
        """Handles near-zero probabilities gracefully."""
        p = torch.tensor([0.999, 0.001])
        q = torch.tensor([0.001, 0.999])

        result = kullback_leibler_divergence(p, q)

        assert torch.isfinite(result)
        assert result > 0

    def test_very_small_probabilities(self):
        """Handles very small probabilities without underflow."""
        p = torch.tensor([1e-10, 1.0 - 1e-10])
        q = torch.tensor([0.5, 0.5])

        result = kullback_leibler_divergence(p, q)

        assert torch.isfinite(result)

    def test_dtype_float32(self):
        """Works with float32 inputs."""
        p = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)

        result = kullback_leibler_divergence(p, q)

        assert result.dtype == torch.float32
        assert torch.isfinite(result)

    def test_dtype_float64(self):
        """Works with float64 inputs."""
        p = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)

        result = kullback_leibler_divergence(p, q)

        assert result.dtype == torch.float64
        assert torch.isfinite(result)

    def test_dtype_promotion(self):
        """Promotes dtypes when inputs differ."""
        p = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)

        result = kullback_leibler_divergence(p, q)

        assert result.dtype == torch.float64

    def test_single_element_distribution(self):
        """Handles degenerate single-element distribution."""
        p = torch.tensor([1.0])
        q = torch.tensor([1.0])

        result = kullback_leibler_divergence(p, q)

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-7)

    def test_large_distribution(self):
        """Handles large distributions efficiently."""
        p = torch.softmax(torch.randn(10000), dim=-1)
        q = torch.softmax(torch.randn(10000), dim=-1)

        result = kullback_leibler_divergence(p, q)

        assert torch.isfinite(result)
        assert result >= 0


class TestKLDivergenceValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Raises error for invalid input_type."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="input_type must be one of"):
            kullback_leibler_divergence(p, q, input_type="invalid")

    def test_invalid_reduction(self):
        """Raises error for invalid reduction."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="reduction must be one of"):
            kullback_leibler_divergence(p, q, reduction="invalid")

    def test_mismatched_sizes(self):
        """Raises error for mismatched distribution sizes."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.3, 0.3, 0.4])

        with pytest.raises(ValueError, match="Distribution sizes must match"):
            kullback_leibler_divergence(p, q)

    def test_invalid_dim(self):
        """Raises error for invalid dim."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(IndexError, match="dim .* out of range"):
            kullback_leibler_divergence(p, q, dim=5)

    def test_non_tensor_input(self):
        """Raises error for non-tensor inputs."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            kullback_leibler_divergence([0.5, 0.5], torch.tensor([0.5, 0.5]))

        with pytest.raises(TypeError, match="must be a Tensor"):
            kullback_leibler_divergence(torch.tensor([0.5, 0.5]), [0.5, 0.5])

    def test_pairwise_requires_2d(self):
        """pairwise=True requires at least 2D inputs."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="pairwise=True requires"):
            kullback_leibler_divergence(p, q, pairwise=True)


class TestKLDivergenceDim:
    """Dimension parameter tests."""

    def test_dim_0(self):
        """Works with dim=0 for distribution along first axis."""
        p = torch.softmax(torch.randn(5, 3), dim=0)  # Each column sums to 1
        q = torch.softmax(torch.randn(5, 3), dim=0)

        result = kullback_leibler_divergence(p, q, dim=0)

        assert result.shape == torch.Size([3])

    def test_dim_negative(self):
        """Works with negative dim values."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(3, 5), dim=-1)

        result_neg = kullback_leibler_divergence(p, q, dim=-1)
        result_pos = kullback_leibler_divergence(p, q, dim=1)

        assert torch.allclose(result_neg, result_pos)

    def test_dim_middle(self):
        """Works with middle dimension."""
        p = torch.softmax(torch.randn(2, 5, 3), dim=1)  # Middle dim sums to 1
        q = torch.softmax(torch.randn(2, 5, 3), dim=1)

        result = kullback_leibler_divergence(p, q, dim=1)

        assert result.shape == torch.Size([2, 3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestKLDivergenceCUDA:
    """CUDA backend tests."""

    def test_cuda_matches_cpu(self):
        """CUDA produces same results as CPU."""
        p_cpu = torch.softmax(torch.randn(10), dim=-1)
        q_cpu = torch.softmax(torch.randn(10), dim=-1)

        p_cuda = p_cpu.cuda()
        q_cuda = q_cpu.cuda()

        result_cpu = kullback_leibler_divergence(p_cpu, q_cpu)
        result_cuda = kullback_leibler_divergence(p_cuda, q_cuda)

        assert torch.isclose(result_cpu, result_cuda.cpu(), rtol=1e-5)

    def test_cuda_batch(self):
        """CUDA handles batched input."""
        p = torch.softmax(torch.randn(100, 50).cuda(), dim=-1)
        q = torch.softmax(torch.randn(100, 50).cuda(), dim=-1)

        result = kullback_leibler_divergence(p, q)

        assert result.device.type == "cuda"
        assert result.shape == torch.Size([100])

    def test_cuda_gradients(self):
        """CUDA gradients are correct."""
        p = torch.softmax(torch.randn(10, dtype=torch.float64).cuda(), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float64).cuda(), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        result = kullback_leibler_divergence(p, q)
        result.backward()

        assert p.grad is not None
        assert q.grad is not None
        assert torch.isfinite(p.grad).all()
        assert torch.isfinite(q.grad).all()

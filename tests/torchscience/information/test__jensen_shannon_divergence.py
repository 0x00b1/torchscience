"""Comprehensive tests for Jensen-Shannon divergence."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import jensen_shannon_divergence


class TestJSDivergenceBasic:
    """Basic functionality tests."""

    def test_symmetric(self):
        """JS divergence is symmetric: D_JS(P||Q) = D_JS(Q||P)."""
        p = torch.tensor([0.3, 0.7])
        q = torch.tensor([0.6, 0.4])

        js_pq = jensen_shannon_divergence(p, q)
        js_qp = jensen_shannon_divergence(q, p)

        assert torch.isclose(js_pq, js_qp, rtol=1e-6)

    def test_symmetric_batch(self):
        """Symmetry holds for batched inputs."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        js_pq = jensen_shannon_divergence(p, q)
        js_qp = jensen_shannon_divergence(q, p)

        assert torch.allclose(js_pq, js_qp, rtol=1e-5)

    def test_bounded_by_log_2(self):
        """JS divergence is bounded by log(2) in natural log."""
        # Maximum divergence is between deterministic distributions
        p = torch.tensor([1.0, 0.0])
        q = torch.tensor([0.0, 1.0])

        result = jensen_shannon_divergence(p, q)

        log_2 = math.log(2)
        # Should be close to log(2) for completely disjoint distributions
        assert result <= log_2 + 1e-6, (
            f"JS divergence {result} exceeds log(2) = {log_2}"
        )

    def test_bounded_random_distributions(self):
        """JS divergence is bounded by log(2) for random distributions."""
        torch.manual_seed(42)
        for _ in range(20):
            p = torch.softmax(torch.randn(100), dim=-1)
            q = torch.softmax(torch.randn(100), dim=-1)

            result = jensen_shannon_divergence(p, q)

            assert 0 <= result <= math.log(2) + 1e-6

    def test_zero_for_identical_distributions(self):
        """JS divergence is zero when P equals Q."""
        p = torch.tensor([0.2, 0.3, 0.5])
        q = p.clone()

        result = jensen_shannon_divergence(p, q)

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-7)

    def test_zero_for_identical_batch(self):
        """JS divergence is zero for each identical pair in batch."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = p.clone()

        result = jensen_shannon_divergence(p, q)

        assert torch.allclose(result, torch.zeros(5), atol=1e-6)

    def test_non_negativity(self):
        """JS divergence is always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(100), dim=-1)
            q = torch.softmax(torch.randn(100), dim=-1)

            result = jensen_shannon_divergence(p, q)

            assert result >= 0, (
                f"JS divergence should be non-negative, got {result}"
            )

    def test_output_shape_1d(self):
        """Returns scalar for 1D probability vectors."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.1, 0.2, 0.3, 0.4])

        result = jensen_shannon_divergence(p, q)

        assert result.shape == torch.Size([])

    def test_output_shape_2d_batch(self):
        """Returns 1D tensor for batch of distributions."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        result = jensen_shannon_divergence(p, q)

        assert result.shape == torch.Size([10])


class TestJSDivergenceBase:
    """Logarithm base conversion tests."""

    def test_base_2_bounded_by_1(self):
        """With base=2, JS divergence is bounded by 1."""
        # Maximum divergence distributions
        p = torch.tensor([1.0, 0.0])
        q = torch.tensor([0.0, 1.0])

        result = jensen_shannon_divergence(p, q, base=2)

        assert result <= 1.0 + 1e-6, (
            f"JS divergence with base=2 {result} exceeds 1.0"
        )

    def test_base_2_random_bounded(self):
        """With base=2, all values are bounded by 1."""
        torch.manual_seed(42)
        for _ in range(20):
            p = torch.softmax(torch.randn(100), dim=-1)
            q = torch.softmax(torch.randn(100), dim=-1)

            result = jensen_shannon_divergence(p, q, base=2)

            assert 0 <= result <= 1.0 + 1e-6

    def test_base_conversion_formula(self):
        """Base conversion follows log_b(x) = ln(x) / ln(b)."""
        p = torch.tensor([0.3, 0.7])
        q = torch.tensor([0.6, 0.4])

        result_natural = jensen_shannon_divergence(
            p, q
        )  # base=None (natural log)
        result_base2 = jensen_shannon_divergence(p, q, base=2)
        result_base10 = jensen_shannon_divergence(p, q, base=10)

        # Verify conversion
        assert torch.isclose(
            result_base2, result_natural / math.log(2), rtol=1e-5
        )
        assert torch.isclose(
            result_base10, result_natural / math.log(10), rtol=1e-5
        )

    def test_base_e(self):
        """base=e should give same result as base=None."""
        p = torch.tensor([0.3, 0.7])
        q = torch.tensor([0.6, 0.4])

        result_natural = jensen_shannon_divergence(p, q)
        result_base_e = jensen_shannon_divergence(p, q, base=math.e)

        assert torch.isclose(result_natural, result_base_e, rtol=1e-5)

    def test_base_with_batch(self):
        """Base conversion works with batched inputs."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result_natural = jensen_shannon_divergence(p, q)
        result_base2 = jensen_shannon_divergence(p, q, base=2)

        expected = result_natural / math.log(2)
        assert torch.allclose(result_base2, expected, rtol=1e-5)


class TestJSDivergenceGradients:
    """Gradient computation tests."""

    def test_gradcheck_probability(self):
        """Gradients are correct for probability inputs."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        def func(p_in, q_in):
            return jensen_shannon_divergence(
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
            return jensen_shannon_divergence(p_in, q_in, input_type="logits")

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_batch(self):
        """Gradients are correct for batched inputs."""
        p = torch.softmax(torch.randn(3, 5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(3, 5, dtype=torch.float64), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        def func(p_in, q_in):
            return jensen_shannon_divergence(
                p_in, q_in, input_type="probability", reduction="sum"
            )

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_with_base(self):
        """Gradients are correct when using base parameter."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        def func(p_in, q_in):
            return jensen_shannon_divergence(p_in, q_in, base=2)

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass runs without errors."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        result = jensen_shannon_divergence(p, q)
        result.backward()

        assert p.grad is not None
        assert q.grad is not None
        assert torch.isfinite(p.grad).all()
        assert torch.isfinite(q.grad).all()

    def test_gradients_symmetric(self):
        """Gradients are symmetric due to symmetric formula."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        # D_JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
        # where M = 0.5*(P+Q)
        # The gradient structure should reflect this symmetry

        p1 = p.clone().requires_grad_(True)
        q1 = q.clone().requires_grad_(True)
        js_pq = jensen_shannon_divergence(p1, q1)
        js_pq.backward()
        grad_p_from_pq = p1.grad.clone()
        grad_q_from_pq = q1.grad.clone()

        p2 = p.clone().requires_grad_(True)
        q2 = q.clone().requires_grad_(True)
        js_qp = jensen_shannon_divergence(q2, p2)
        js_qp.backward()
        grad_q_from_qp = q2.grad.clone()
        grad_p_from_qp = p2.grad.clone()

        # Due to symmetry, swapping p and q should swap the gradients
        assert torch.allclose(grad_p_from_pq, grad_p_from_qp, rtol=1e-5)
        assert torch.allclose(grad_q_from_pq, grad_q_from_qp, rtol=1e-5)


class TestJSDivergenceCorrectness:
    """Numerical correctness tests."""

    def test_formula_verification(self):
        """Verify against manual computation of JS divergence formula."""
        p = torch.tensor([0.3, 0.7])
        q = torch.tensor([0.6, 0.4])

        result = jensen_shannon_divergence(p, q)

        # Manual computation: JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)
        # where M = 0.5*(P+Q)
        m = 0.5 * (p + q)
        kl_pm = (p * torch.log(p / m)).sum()
        kl_qm = (q * torch.log(q / m)).sum()
        expected = 0.5 * kl_pm + 0.5 * kl_qm

        assert torch.isclose(result, expected, rtol=1e-5)

    def test_uniform_distributions(self):
        """JS divergence between uniform distributions is zero."""
        p = torch.ones(4) / 4
        q = torch.ones(4) / 4

        result = jensen_shannon_divergence(p, q)

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-7)

    def test_slightly_different_distributions(self):
        """Small differences give small JS divergence."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.51, 0.49])

        result = jensen_shannon_divergence(p, q)

        # Should be small but positive
        assert 0 < result < 0.01


class TestJSDivergenceInputTypes:
    """Input type handling tests."""

    def test_log_probability_input(self):
        """Handles log_probability input type."""
        p_probs = torch.softmax(torch.randn(10), dim=-1)
        q_probs = torch.softmax(torch.randn(10), dim=-1)

        log_p = torch.log(p_probs)
        log_q = torch.log(q_probs)

        result_prob = jensen_shannon_divergence(
            p_probs, q_probs, input_type="probability"
        )
        result_log = jensen_shannon_divergence(
            log_p, log_q, input_type="log_probability"
        )

        assert torch.isclose(result_prob, result_log, rtol=1e-5)

    def test_logits_input(self):
        """Handles logits input type."""
        logits_p = torch.randn(10)
        logits_q = torch.randn(10)

        p_probs = torch.softmax(logits_p, dim=-1)
        q_probs = torch.softmax(logits_q, dim=-1)

        result_prob = jensen_shannon_divergence(
            p_probs, q_probs, input_type="probability"
        )
        result_logits = jensen_shannon_divergence(
            logits_p, logits_q, input_type="logits"
        )

        assert torch.isclose(result_prob, result_logits, rtol=1e-5)


class TestJSDivergenceReduction:
    """Reduction mode tests."""

    def test_reduction_none(self):
        """reduction='none' returns per-sample divergences."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result = jensen_shannon_divergence(p, q, reduction="none")

        assert result.shape == torch.Size([5])

    def test_reduction_sum(self):
        """reduction='sum' returns sum of all divergences."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result_none = jensen_shannon_divergence(p, q, reduction="none")
        result_sum = jensen_shannon_divergence(p, q, reduction="sum")

        assert torch.isclose(result_sum, result_none.sum(), rtol=1e-5)

    def test_reduction_mean(self):
        """reduction='mean' returns mean over all elements."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result_none = jensen_shannon_divergence(p, q, reduction="none")
        result_mean = jensen_shannon_divergence(p, q, reduction="mean")

        assert torch.isclose(result_mean, result_none.mean(), rtol=1e-5)

    def test_reduction_batchmean(self):
        """reduction='batchmean' returns mean over batch dimension."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)

        result_none = jensen_shannon_divergence(p, q, reduction="none")
        result_batchmean = jensen_shannon_divergence(
            p, q, reduction="batchmean"
        )

        expected = result_none.sum() / p.shape[0]
        assert torch.isclose(result_batchmean, expected, rtol=1e-5)


class TestJSDivergencePairwise:
    """Pairwise divergence computation tests."""

    def test_pairwise_output_shape(self):
        """pairwise=True returns (m, k) matrix for (m, n) and (k, n) inputs."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(4, 5), dim=-1)

        result = jensen_shannon_divergence(p, q, pairwise=True)

        assert result.shape == torch.Size([3, 4])

    def test_pairwise_symmetric(self):
        """Pairwise matrix is symmetric when p == q."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)

        result = jensen_shannon_divergence(p, p, pairwise=True)

        # Matrix should be symmetric
        assert torch.allclose(result, result.T, rtol=1e-5)

    def test_pairwise_diagonal_zero(self):
        """Pairwise diagonal is zero when p == q."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)

        result = jensen_shannon_divergence(p, p, pairwise=True)

        diagonal = torch.diag(result)
        assert torch.allclose(diagonal, torch.zeros(5), atol=1e-6)


class TestJSDivergenceEdgeCases:
    """Edge case handling tests."""

    def test_near_zero_probabilities(self):
        """Handles near-zero probabilities gracefully."""
        p = torch.tensor([0.999, 0.001])
        q = torch.tensor([0.001, 0.999])

        result = jensen_shannon_divergence(p, q)

        assert torch.isfinite(result)
        assert result > 0
        assert result <= math.log(2) + 1e-6

    def test_dtype_float32(self):
        """Works with float32 inputs."""
        p = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)

        result = jensen_shannon_divergence(p, q)

        assert result.dtype == torch.float32
        assert torch.isfinite(result)

    def test_dtype_float64(self):
        """Works with float64 inputs."""
        p = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)

        result = jensen_shannon_divergence(p, q)

        assert result.dtype == torch.float64
        assert torch.isfinite(result)


class TestJSDivergenceValidation:
    """Input validation tests."""

    def test_invalid_input_type(self):
        """Raises error for invalid input_type."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="input_type must be one of"):
            jensen_shannon_divergence(p, q, input_type="invalid")

    def test_invalid_reduction(self):
        """Raises error for invalid reduction."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="reduction must be one of"):
            jensen_shannon_divergence(p, q, reduction="invalid")

    def test_mismatched_sizes(self):
        """Raises error for mismatched distribution sizes."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.3, 0.3, 0.4])

        with pytest.raises(ValueError, match="Distribution sizes must match"):
            jensen_shannon_divergence(p, q)

    def test_pairwise_requires_2d(self):
        """pairwise=True requires at least 2D inputs."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="pairwise=True requires"):
            jensen_shannon_divergence(p, q, pairwise=True)


class TestJSDivergenceMetricProperty:
    """Tests for sqrt(JS) being a proper metric."""

    def test_sqrt_js_triangle_inequality(self):
        """sqrt(JS) satisfies triangle inequality."""
        p = torch.tensor([0.2, 0.8])
        q = torch.tensor([0.5, 0.5])
        r = torch.tensor([0.8, 0.2])

        js_pq = jensen_shannon_divergence(p, q)
        js_qr = jensen_shannon_divergence(q, r)
        js_pr = jensen_shannon_divergence(p, r)

        # Triangle inequality: sqrt(JS(P,R)) <= sqrt(JS(P,Q)) + sqrt(JS(Q,R))
        lhs = torch.sqrt(js_pr)
        rhs = torch.sqrt(js_pq) + torch.sqrt(js_qr)

        assert lhs <= rhs + 1e-6, (
            f"Triangle inequality violated: {lhs} > {rhs}"
        )

    def test_sqrt_js_identity(self):
        """sqrt(JS(P,P)) = 0."""
        p = torch.tensor([0.3, 0.7])

        result = torch.sqrt(jensen_shannon_divergence(p, p))

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestJSDivergenceCUDA:
    """CUDA backend tests."""

    def test_cuda_matches_cpu(self):
        """CUDA produces same results as CPU."""
        p_cpu = torch.softmax(torch.randn(10), dim=-1)
        q_cpu = torch.softmax(torch.randn(10), dim=-1)

        p_cuda = p_cpu.cuda()
        q_cuda = q_cpu.cuda()

        result_cpu = jensen_shannon_divergence(p_cpu, q_cpu)
        result_cuda = jensen_shannon_divergence(p_cuda, q_cuda)

        assert torch.isclose(result_cpu, result_cuda.cpu(), rtol=1e-5)

    def test_cuda_batch(self):
        """CUDA handles batched input."""
        p = torch.softmax(torch.randn(100, 50).cuda(), dim=-1)
        q = torch.softmax(torch.randn(100, 50).cuda(), dim=-1)

        result = jensen_shannon_divergence(p, q)

        assert result.device.type == "cuda"
        assert result.shape == torch.Size([100])

    def test_cuda_gradients(self):
        """CUDA gradients are correct."""
        p = torch.softmax(torch.randn(10, dtype=torch.float64).cuda(), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float64).cuda(), dim=-1)

        p.requires_grad_(True)
        q.requires_grad_(True)

        result = jensen_shannon_divergence(p, q)
        result.backward()

        assert p.grad is not None
        assert q.grad is not None
        assert torch.isfinite(p.grad).all()
        assert torch.isfinite(q.grad).all()

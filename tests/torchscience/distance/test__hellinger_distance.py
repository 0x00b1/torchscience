"""Comprehensive tests for Hellinger distance."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.distance import hellinger_distance


class TestHellingerDistanceBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D probability vectors."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.1, 0.2, 0.3, 0.4])
        result = hellinger_distance(p, q)
        assert result.shape == torch.Size([])

    def test_output_shape_batch(self):
        """Returns correct shape for batched input."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)
        result = hellinger_distance(p, q)
        assert result.shape == torch.Size([10])


class TestHellingerDistanceCorrectness:
    """Numerical correctness tests."""

    def test_zero_for_identical(self):
        """Hellinger distance is zero for identical distributions."""
        p = torch.softmax(torch.randn(10), dim=-1)
        result = hellinger_distance(p, p)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_symmetric(self):
        """Hellinger distance is symmetric: H(P,Q) = H(Q,P)."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        h_pq = hellinger_distance(p, q)
        h_qp = hellinger_distance(q, p)

        assert torch.isclose(h_pq, h_qp, rtol=1e-5)

    def test_bounds(self):
        """0 <= H(P,Q) <= 1."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(10), dim=-1)
            q = torch.softmax(torch.randn(10), dim=-1)

            h = hellinger_distance(p, q)

            assert h >= -1e-6, f"Expected H >= 0, got {h}"
            assert h <= 1 + 1e-6, f"Expected H <= 1, got {h}"

    def test_maximum_distance(self):
        """Non-overlapping distributions have H = 1."""
        p = torch.tensor([1.0, 0.0])
        q = torch.tensor([0.0, 1.0])

        h = hellinger_distance(p, q)

        # Due to epsilon clamping, H is very close to 1 but not exactly
        assert torch.isclose(h, torch.tensor(1.0), atol=1e-3)

    def test_manual_computation(self):
        """Verify against manual computation."""
        p = torch.tensor([0.4, 0.6])
        q = torch.tensor([0.5, 0.5])

        result = hellinger_distance(p, q)

        # Manual: H = (1/sqrt(2)) * sqrt((sqrt(0.4)-sqrt(0.5))^2 + (sqrt(0.6)-sqrt(0.5))^2)
        sqrt_p = torch.sqrt(p)
        sqrt_q = torch.sqrt(q)
        expected = torch.sqrt(((sqrt_p - sqrt_q) ** 2).sum() / 2)

        assert torch.isclose(result, expected, rtol=1e-5)


class TestHellingerDistanceGradients:
    """Gradient computation tests."""

    def test_gradcheck(self):
        """Gradients are correct."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)
        q.requires_grad_(True)

        def func(p_in, q_in):
            return hellinger_distance(p_in, q_in)

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_runs(self):
        """Backward pass runs without errors."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)
        p.requires_grad_(True)
        q.requires_grad_(True)

        result = hellinger_distance(p, q)
        result.backward()

        assert p.grad is not None
        assert q.grad is not None
        assert torch.isfinite(p.grad).all()
        assert torch.isfinite(q.grad).all()


class TestHellingerDistancePairwise:
    """Pairwise computation tests."""

    def test_pairwise_output_shape(self):
        """pairwise=True returns (m, k) matrix."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(4, 5), dim=-1)

        result = hellinger_distance(p, q, pairwise=True)

        assert result.shape == torch.Size([3, 4])

    def test_pairwise_symmetric(self):
        """Pairwise matrix is symmetric when p == q."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)

        result = hellinger_distance(p, p, pairwise=True)

        assert torch.allclose(result, result.T, rtol=1e-5)


class TestHellingerDistanceValidation:
    """Input validation tests."""

    def test_non_tensor_input(self):
        """Raises error for non-tensor inputs."""
        with pytest.raises(TypeError):
            hellinger_distance([0.5, 0.5], torch.tensor([0.5, 0.5]))

    def test_mismatched_sizes(self):
        """Raises error for mismatched distribution sizes."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.3, 0.3, 0.4])
        with pytest.raises(ValueError, match="Distribution sizes must match"):
            hellinger_distance(p, q)


class TestHellingerDistanceInputTypes:
    """Test different input type handling."""

    def test_log_probability_input(self):
        """Works with log probability inputs."""
        p_prob = torch.softmax(torch.randn(5), dim=-1)
        q_prob = torch.softmax(torch.randn(5), dim=-1)

        h_prob = hellinger_distance(p_prob, q_prob, input_type="probability")
        h_log = hellinger_distance(
            torch.log(p_prob), torch.log(q_prob), input_type="log_probability"
        )

        assert torch.isclose(h_prob, h_log, rtol=1e-4)

    def test_logits_input(self):
        """Works with logits inputs."""
        logits_p = torch.randn(5)
        logits_q = torch.randn(5)

        h_logits = hellinger_distance(logits_p, logits_q, input_type="logits")
        h_prob = hellinger_distance(
            torch.softmax(logits_p, dim=-1),
            torch.softmax(logits_q, dim=-1),
            input_type="probability",
        )

        assert torch.isclose(h_logits, h_prob, rtol=1e-4)


class TestHellingerDistanceReduction:
    """Test reduction modes."""

    def test_reduction_none(self):
        """reduction='none' returns per-sample distances."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        result = hellinger_distance(p, q, reduction="none")

        assert result.shape == torch.Size([10])

    def test_reduction_mean(self):
        """reduction='mean' returns scalar."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        result = hellinger_distance(p, q, reduction="mean")

        assert result.shape == torch.Size([])

    def test_reduction_sum(self):
        """reduction='sum' returns scalar."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        result = hellinger_distance(p, q, reduction="sum")

        assert result.shape == torch.Size([])

    def test_reduction_consistency(self):
        """Reductions are consistent with manual computation."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        none_result = hellinger_distance(p, q, reduction="none")
        mean_result = hellinger_distance(p, q, reduction="mean")
        sum_result = hellinger_distance(p, q, reduction="sum")

        assert torch.isclose(mean_result, none_result.mean(), rtol=1e-5)
        assert torch.isclose(sum_result, none_result.sum(), rtol=1e-5)

"""Comprehensive tests for total variation distance."""

import pytest
import torch

from torchscience.similarity_measure import total_variation_distance


class TestTotalVariationDistanceBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D probability vectors."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.1, 0.2, 0.3, 0.4])
        result = total_variation_distance(p, q)
        assert result.shape == torch.Size([])

    def test_output_shape_batch(self):
        """Returns correct shape for batched input."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)
        result = total_variation_distance(p, q)
        assert result.shape == torch.Size([10])


class TestTotalVariationDistanceCorrectness:
    """Numerical correctness tests."""

    def test_zero_for_identical(self):
        """TV distance is zero for identical distributions."""
        p = torch.softmax(torch.randn(10), dim=-1)
        result = total_variation_distance(p, p)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_symmetric(self):
        """TV distance is symmetric: TV(P,Q) = TV(Q,P)."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        tv_pq = total_variation_distance(p, q)
        tv_qp = total_variation_distance(q, p)

        assert torch.isclose(tv_pq, tv_qp, rtol=1e-5)

    def test_bounds(self):
        """0 <= TV(P,Q) <= 1."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(10), dim=-1)
            q = torch.softmax(torch.randn(10), dim=-1)

            tv = total_variation_distance(p, q)

            assert tv >= -1e-6, f"Expected TV >= 0, got {tv}"
            assert tv <= 1 + 1e-6, f"Expected TV <= 1, got {tv}"

    def test_maximum_distance(self):
        """Non-overlapping distributions have TV = 1."""
        p = torch.tensor([1.0, 0.0])
        q = torch.tensor([0.0, 1.0])

        tv = total_variation_distance(p, q)

        assert torch.isclose(tv, torch.tensor(1.0), rtol=1e-5)

    def test_manual_computation(self):
        """Verify against manual computation."""
        p = torch.tensor([0.4, 0.6])
        q = torch.tensor([0.5, 0.5])

        result = total_variation_distance(p, q)

        # Manual: TV = 0.5 * (|0.4-0.5| + |0.6-0.5|) = 0.5 * (0.1 + 0.1) = 0.1
        expected = torch.tensor(0.1)

        assert torch.isclose(result, expected, rtol=1e-5)


class TestTotalVariationDistanceGradients:
    """Gradient computation tests."""

    def test_backward_runs(self):
        """Backward pass runs without errors."""
        # Use distributions with clear separation to avoid subgradient issues
        p = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
        q = torch.tensor([0.4, 0.3, 0.2, 0.1], requires_grad=True)

        result = total_variation_distance(p, q)
        result.backward()

        assert p.grad is not None
        assert q.grad is not None
        assert torch.isfinite(p.grad).all()
        assert torch.isfinite(q.grad).all()

    def test_gradient_sign(self):
        """Gradients have correct sign."""
        p = torch.tensor([0.3, 0.7], requires_grad=True)
        q = torch.tensor([0.5, 0.5], requires_grad=True)

        result = total_variation_distance(p, q)
        result.backward()

        # d/dp_i = 0.5 * sign(p_i - q_i)
        # p[0] - q[0] = 0.3 - 0.5 = -0.2 < 0, so sign = -1, grad = -0.5
        # p[1] - q[1] = 0.7 - 0.5 = 0.2 > 0, so sign = +1, grad = +0.5
        assert p.grad[0] < 0  # negative gradient
        assert p.grad[1] > 0  # positive gradient


class TestTotalVariationDistancePairwise:
    """Pairwise computation tests."""

    def test_pairwise_output_shape(self):
        """pairwise=True returns (m, k) matrix."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(4, 5), dim=-1)

        result = total_variation_distance(p, q, pairwise=True)

        assert result.shape == torch.Size([3, 4])

    def test_pairwise_symmetric(self):
        """Pairwise matrix is symmetric when p == q."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)

        result = total_variation_distance(p, p, pairwise=True)

        assert torch.allclose(result, result.T, rtol=1e-5)


class TestTotalVariationDistanceValidation:
    """Input validation tests."""

    def test_non_tensor_input(self):
        """Raises error for non-tensor inputs."""
        with pytest.raises(TypeError):
            total_variation_distance([0.5, 0.5], torch.tensor([0.5, 0.5]))

    def test_mismatched_sizes(self):
        """Raises error for mismatched distribution sizes."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.3, 0.3, 0.4])
        with pytest.raises(ValueError, match="Distribution sizes must match"):
            total_variation_distance(p, q)

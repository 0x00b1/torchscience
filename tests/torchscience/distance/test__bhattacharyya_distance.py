"""Comprehensive tests for Bhattacharyya distance."""

import pytest
import torch

from torchscience.distance import bhattacharyya_distance


class TestBhattacharyyaDistanceBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D probability vectors."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.1, 0.2, 0.3, 0.4])
        result = bhattacharyya_distance(p, q)
        assert result.shape == torch.Size([])

    def test_output_shape_batch(self):
        """Returns correct shape for batched input."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)
        result = bhattacharyya_distance(p, q)
        assert result.shape == torch.Size([10])


class TestBhattacharyyaDistanceCorrectness:
    """Numerical correctness tests."""

    def test_zero_for_identical(self):
        """Bhattacharyya distance is zero for identical distributions."""
        p = torch.softmax(torch.randn(10), dim=-1)
        result = bhattacharyya_distance(p, p)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_symmetric(self):
        """Bhattacharyya distance is symmetric: D_B(P,Q) = D_B(Q,P)."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        db_pq = bhattacharyya_distance(p, q)
        db_qp = bhattacharyya_distance(q, p)

        assert torch.isclose(db_pq, db_qp, rtol=1e-5)

    def test_non_negative(self):
        """Bhattacharyya distance is non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(10), dim=-1)
            q = torch.softmax(torch.randn(10), dim=-1)

            db = bhattacharyya_distance(p, q)

            assert db >= -1e-6, f"Expected D_B >= 0, got {db}"

    def test_large_for_non_overlapping(self):
        """Non-overlapping distributions have large (infinite) distance."""
        p = torch.tensor([1.0, 0.0])
        q = torch.tensor([0.0, 1.0])

        db = bhattacharyya_distance(p, q)

        # BC = sqrt(eps)*sqrt(eps) = eps for non-overlapping, so D_B = -ln(eps) is large
        # With eps ~1e-3 to 1e-4, expect D_B ~7-9
        assert db > 5.0, (
            f"Expected large distance for non-overlapping, got {db}"
        )

    def test_manual_computation(self):
        """Verify against manual computation."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        result = bhattacharyya_distance(p, q)

        # BC = sqrt(0.5*0.5) + sqrt(0.5*0.5) = 0.5 + 0.5 = 1.0
        # D_B = -ln(1.0) = 0.0
        expected = torch.tensor(0.0)

        assert torch.isclose(result, expected, atol=1e-6)

    def test_relationship_with_hellinger(self):
        """Verify relationship: H^2 = 1 - exp(-D_B)."""
        from torchscience.distance import hellinger_distance

        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        h = hellinger_distance(p, q)
        db = bhattacharyya_distance(p, q)

        # H^2 = 1 - exp(-D_B)
        h_squared_from_db = 1 - torch.exp(-db)

        assert torch.isclose(h**2, h_squared_from_db, rtol=1e-4)


class TestBhattacharyyaDistanceGradients:
    """Gradient computation tests."""

    def test_backward_runs(self):
        """Backward pass runs without errors."""
        p = torch.softmax(torch.randn(10), dim=-1).requires_grad_(True)
        q = torch.softmax(torch.randn(10), dim=-1).requires_grad_(True)

        result = bhattacharyya_distance(p, q)
        result.backward()

        assert p.grad is not None
        assert q.grad is not None
        assert torch.isfinite(p.grad).all()
        assert torch.isfinite(q.grad).all()

    def test_gradient_sign(self):
        """Gradients have correct sign."""
        # For D_B = -ln(BC), increasing overlap should decrease distance
        # dD_B/dp_j = -sqrt(q_j) / (2 * BC * sqrt(p_j))
        # This is always negative (for positive p, q)
        p = torch.tensor([0.3, 0.7], requires_grad=True)
        q = torch.tensor([0.5, 0.5])

        result = bhattacharyya_distance(p, q)
        result.backward()

        # All gradients should be negative since increasing p increases BC, decreasing D_B
        assert (p.grad < 0).all()

    def test_gradcheck(self):
        """Numerical gradient check."""
        p = torch.softmax(
            torch.randn(5, dtype=torch.float64), dim=-1
        ).requires_grad_(True)
        q = torch.softmax(
            torch.randn(5, dtype=torch.float64), dim=-1
        ).requires_grad_(True)

        # Use less strict tolerance due to numerical sensitivity
        assert torch.autograd.gradcheck(
            bhattacharyya_distance,
            (p, q),
            eps=1e-5,
            atol=1e-4,
            rtol=1e-3,
        )


class TestBhattacharyyaDistancePairwise:
    """Pairwise computation tests."""

    def test_pairwise_output_shape(self):
        """pairwise=True returns (m, k) matrix."""
        p = torch.softmax(torch.randn(3, 5), dim=-1)
        q = torch.softmax(torch.randn(4, 5), dim=-1)

        result = bhattacharyya_distance(p, q, pairwise=True)

        assert result.shape == torch.Size([3, 4])

    def test_pairwise_symmetric(self):
        """Pairwise matrix is symmetric when p == q."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)

        result = bhattacharyya_distance(p, p, pairwise=True)

        assert torch.allclose(result, result.T, rtol=1e-5)

    def test_pairwise_diagonal_zero(self):
        """Diagonal of pairwise matrix is zero when p == q."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)

        result = bhattacharyya_distance(p, p, pairwise=True)

        assert torch.allclose(result.diag(), torch.zeros(5), atol=1e-5)


class TestBhattacharyyaDistanceValidation:
    """Input validation tests."""

    def test_non_tensor_input(self):
        """Raises error for non-tensor inputs."""
        with pytest.raises(TypeError):
            bhattacharyya_distance([0.5, 0.5], torch.tensor([0.5, 0.5]))

    def test_mismatched_sizes(self):
        """Raises error for mismatched distribution sizes."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.3, 0.3, 0.4])
        with pytest.raises(ValueError, match="Distribution sizes must match"):
            bhattacharyya_distance(p, q)

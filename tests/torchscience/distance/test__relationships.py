"""Tests verifying mathematical relationships between distance functions."""

import pytest
import torch

from torchscience.distance import (
    bhattacharyya_distance,
    hellinger_distance,
    total_variation_distance,
)


class TestHellingerBhattacharyyaRelationship:
    """Verify H^2(P,Q) = 1 - exp(-D_B(P,Q))."""

    def test_relationship(self):
        """Hellinger and Bhattacharyya are related."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        h = hellinger_distance(p, q)
        d_b = bhattacharyya_distance(p, q)

        # H^2 = 1 - exp(-D_B)
        assert torch.isclose(h**2, 1 - torch.exp(-d_b), rtol=1e-4)

    def test_relationship_batch(self):
        """Relationship holds for batched inputs."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)
        q = torch.softmax(torch.randn(5, 8), dim=-1)

        h = hellinger_distance(p, q)
        d_b = bhattacharyya_distance(p, q)

        assert torch.allclose(h**2, 1 - torch.exp(-d_b), rtol=1e-4)


class TestDistanceInequalities:
    """Verify known inequalities between distances."""

    def test_tv_hellinger_inequality(self):
        """TV(P,Q)^2 <= 2 * H(P,Q)^2."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(10), dim=-1)
            q = torch.softmax(torch.randn(10), dim=-1)

            tv = total_variation_distance(p, q)
            h = hellinger_distance(p, q)

            assert tv**2 <= 2 * h**2 + 1e-5

    def test_hellinger_tv_inequality(self):
        """H(P,Q)^2 <= TV(P,Q)."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(10), dim=-1)
            q = torch.softmax(torch.randn(10), dim=-1)

            tv = total_variation_distance(p, q)
            h = hellinger_distance(p, q)

            assert h**2 <= tv + 1e-5


class TestSymmetry:
    """All distances should be symmetric."""

    @pytest.mark.parametrize(
        "distance_fn",
        [
            hellinger_distance,
            bhattacharyya_distance,
            total_variation_distance,
        ],
    )
    def test_symmetry(self, distance_fn):
        """Distance is symmetric: D(P,Q) = D(Q,P)."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        d_pq = distance_fn(p, q)
        d_qp = distance_fn(q, p)

        assert torch.isclose(d_pq, d_qp, rtol=1e-5)

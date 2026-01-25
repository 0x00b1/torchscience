"""Tests for edge_sample."""

import pytest
import torch

from torchscience.geometry.intersection._edge_sample import edge_sample
from torchscience.geometry.intersection._edge_sampling_result import (
    EdgeSamples,
    SilhouetteEdges,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def triangle_vertices():
    """A simple triangle in 3D."""
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ]
    )


@pytest.fixture()
def two_edge_silhouette():
    """Two silhouette edges from the triangle."""
    return SilhouetteEdges(
        edge_indices=torch.tensor([0, 1]),
        edges=torch.tensor([[0, 1], [1, 2]]),
        front_face=torch.tensor([0, 0]),
        back_face=torch.tensor([-1, -1]),
        batch_size=[2],
    )


@pytest.fixture()
def single_edge_silhouette():
    """Single silhouette edge."""
    return SilhouetteEdges(
        edge_indices=torch.tensor([0]),
        edges=torch.tensor([[0, 1]]),
        front_face=torch.tensor([0]),
        back_face=torch.tensor([-1]),
        batch_size=[1],
    )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestEdgeSampleShapes:
    """Output shape validation."""

    def test_single_sample_per_edge(
        self, triangle_vertices, two_edge_silhouette
    ):
        """Default num_samples_per_edge=1 gives num_sil total samples."""
        result = edge_sample(triangle_vertices, two_edge_silhouette)
        assert isinstance(result, EdgeSamples)
        assert result.positions.shape == (2, 3)
        assert result.edge_indices.shape == (2,)
        assert result.parametric_t.shape == (2,)
        assert result.edge_tangent.shape == (2, 3)
        assert result.edge_length.shape == (2,)

    def test_multiple_samples_per_edge(
        self, triangle_vertices, two_edge_silhouette
    ):
        """num_samples_per_edge=4 gives 2*4=8 total samples."""
        K = 4
        result = edge_sample(
            triangle_vertices, two_edge_silhouette, num_samples_per_edge=K
        )
        assert result.positions.shape == (8, 3)
        assert result.edge_indices.shape == (8,)
        assert result.parametric_t.shape == (8,)
        assert result.edge_tangent.shape == (8, 3)
        assert result.edge_length.shape == (8,)

    def test_single_edge_many_samples(
        self, triangle_vertices, single_edge_silhouette
    ):
        """Single edge with many samples."""
        K = 16
        result = edge_sample(
            triangle_vertices, single_edge_silhouette, num_samples_per_edge=K
        )
        assert result.positions.shape == (16, 3)


# ---------------------------------------------------------------------------
# Parametric t tests
# ---------------------------------------------------------------------------


class TestEdgeSampleParametricT:
    """Parametric t value properties."""

    def test_t_in_unit_interval(self, triangle_vertices, two_edge_silhouette):
        """All parametric t values must be in [0, 1]."""
        result = edge_sample(
            triangle_vertices, two_edge_silhouette, num_samples_per_edge=32
        )
        assert (result.parametric_t >= 0.0).all()
        assert (result.parametric_t <= 1.0).all()

    def test_stratified_samples_in_correct_strata(
        self, triangle_vertices, single_edge_silhouette
    ):
        """Stratified samples must lie in their assigned stratum."""
        K = 8
        result = edge_sample(
            triangle_vertices,
            single_edge_silhouette,
            num_samples_per_edge=K,
            stratified=True,
        )
        t = result.parametric_t  # (K,) since single edge
        for k in range(K):
            lo = k / K
            hi = (k + 1) / K
            assert t[k] >= lo, f"stratum {k}: t={t[k].item()} < lo={lo}"
            assert t[k] <= hi, f"stratum {k}: t={t[k].item()} > hi={hi}"

    def test_uniform_samples_in_unit_interval(
        self, triangle_vertices, two_edge_silhouette
    ):
        """Uniform (non-stratified) samples are in [0, 1]."""
        result = edge_sample(
            triangle_vertices,
            two_edge_silhouette,
            num_samples_per_edge=64,
            stratified=False,
        )
        assert (result.parametric_t >= 0.0).all()
        assert (result.parametric_t <= 1.0).all()


# ---------------------------------------------------------------------------
# Position interpolation tests
# ---------------------------------------------------------------------------


class TestEdgeSamplePositions:
    """Verify positions lie on edges and match manual interpolation."""

    def test_positions_on_edge(
        self, triangle_vertices, single_edge_silhouette
    ):
        """Positions must equal (1-t)*vi + t*vj for edge (0,1)."""
        K = 8
        result = edge_sample(
            triangle_vertices,
            single_edge_silhouette,
            num_samples_per_edge=K,
        )
        v0 = triangle_vertices[0]  # (3,)
        v1 = triangle_vertices[1]  # (3,)
        t = result.parametric_t.unsqueeze(-1)  # (K, 1)
        expected = (1.0 - t) * v0 + t * v1
        assert torch.allclose(result.positions, expected, atol=1e-6)

    def test_positions_on_two_edges(
        self, triangle_vertices, two_edge_silhouette
    ):
        """Verify positions for two different edges."""
        K = 4
        result = edge_sample(
            triangle_vertices,
            two_edge_silhouette,
            num_samples_per_edge=K,
        )
        edges = two_edge_silhouette.edges
        for sil_idx in range(2):
            vi = triangle_vertices[edges[sil_idx, 0]]
            vj = triangle_vertices[edges[sil_idx, 1]]
            sl = slice(sil_idx * K, (sil_idx + 1) * K)
            t = result.parametric_t[sl].unsqueeze(-1)
            expected = (1.0 - t) * vi + t * vj
            assert torch.allclose(result.positions[sl], expected, atol=1e-6), (
                f"edge {sil_idx} positions mismatch"
            )


# ---------------------------------------------------------------------------
# Tangent and length tests
# ---------------------------------------------------------------------------


class TestEdgeSampleTangentAndLength:
    """Edge tangent and length verification."""

    def test_tangent_is_unit_length(
        self, triangle_vertices, two_edge_silhouette
    ):
        """Edge tangent vectors must be unit length."""
        result = edge_sample(
            triangle_vertices, two_edge_silhouette, num_samples_per_edge=4
        )
        norms = torch.linalg.norm(result.edge_tangent, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_tangent_direction(
        self, triangle_vertices, single_edge_silhouette
    ):
        """Tangent of edge (0,1) should be [1,0,0]."""
        result = edge_sample(triangle_vertices, single_edge_silhouette)
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(result.edge_tangent, expected, atol=1e-6)

    def test_edge_length_positive(
        self, triangle_vertices, two_edge_silhouette
    ):
        """All edge lengths must be positive."""
        result = edge_sample(
            triangle_vertices, two_edge_silhouette, num_samples_per_edge=4
        )
        assert (result.edge_length > 0).all()

    def test_edge_length_correct(
        self, triangle_vertices, single_edge_silhouette
    ):
        """Edge (0,1) from [0,0,0] to [1,0,0] has length 1."""
        result = edge_sample(triangle_vertices, single_edge_silhouette)
        assert torch.isclose(
            result.edge_length, torch.tensor([1.0]), atol=1e-6
        ).all()

    def test_edge_length_correct_second_edge(self, triangle_vertices):
        """Edge (1,2) from [1,0,0] to [0.5,1,0] has correct length."""
        sil = SilhouetteEdges(
            edge_indices=torch.tensor([0]),
            edges=torch.tensor([[1, 2]]),
            front_face=torch.tensor([0]),
            back_face=torch.tensor([-1]),
            batch_size=[1],
        )
        result = edge_sample(triangle_vertices, sil)
        v1 = triangle_vertices[1]
        v2 = triangle_vertices[2]
        expected_len = torch.linalg.norm(v2 - v1)
        assert torch.isclose(result.edge_length[0], expected_len, atol=1e-6)


# ---------------------------------------------------------------------------
# Edge indices tests
# ---------------------------------------------------------------------------


class TestEdgeSampleEdgeIndices:
    """Verify parent edge index bookkeeping."""

    def test_edge_indices_range(self, triangle_vertices, two_edge_silhouette):
        """Edge indices should be in [0, num_sil)."""
        K = 4
        result = edge_sample(
            triangle_vertices, two_edge_silhouette, num_samples_per_edge=K
        )
        assert (result.edge_indices >= 0).all()
        assert (result.edge_indices < 2).all()

    def test_edge_indices_ordering(
        self, triangle_vertices, two_edge_silhouette
    ):
        """First K samples belong to edge 0, next K to edge 1."""
        K = 4
        result = edge_sample(
            triangle_vertices, two_edge_silhouette, num_samples_per_edge=K
        )
        assert (result.edge_indices[:K] == 0).all()
        assert (result.edge_indices[K:] == 1).all()


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


class TestEdgeSampleReproducibility:
    """Random seed reproducibility."""

    def test_same_seed_same_output(
        self, triangle_vertices, two_edge_silhouette
    ):
        """Same generator seed produces identical results."""
        g1 = torch.Generator().manual_seed(42)
        r1 = edge_sample(
            triangle_vertices,
            two_edge_silhouette,
            num_samples_per_edge=8,
            generator=g1,
        )

        g2 = torch.Generator().manual_seed(42)
        r2 = edge_sample(
            triangle_vertices,
            two_edge_silhouette,
            num_samples_per_edge=8,
            generator=g2,
        )

        assert torch.equal(r1.positions, r2.positions)
        assert torch.equal(r1.parametric_t, r2.parametric_t)

    def test_different_seed_different_output(
        self, triangle_vertices, two_edge_silhouette
    ):
        """Different seeds should produce different results."""
        g1 = torch.Generator().manual_seed(42)
        r1 = edge_sample(
            triangle_vertices,
            two_edge_silhouette,
            num_samples_per_edge=8,
            generator=g1,
        )

        g2 = torch.Generator().manual_seed(99)
        r2 = edge_sample(
            triangle_vertices,
            two_edge_silhouette,
            num_samples_per_edge=8,
            generator=g2,
        )

        assert not torch.equal(r1.parametric_t, r2.parametric_t)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


class TestEdgeSampleGradients:
    """Autograd gradient flow through linear interpolation."""

    def test_gradcheck_positions_wrt_vertices(self):
        """torch.autograd.gradcheck for positions w.r.t. vertices."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        sil = SilhouetteEdges(
            edge_indices=torch.tensor([0, 1]),
            edges=torch.tensor([[0, 1], [1, 2]]),
            front_face=torch.tensor([0, 0]),
            back_face=torch.tensor([-1, -1]),
            batch_size=[2],
        )

        # Use a fixed generator so the random t values are deterministic
        # across the finite-difference evaluations in gradcheck.
        # We need to create a wrapper that re-seeds each call.
        def func(verts):
            g = torch.Generator().manual_seed(123)
            result = edge_sample(
                verts,
                sil,
                num_samples_per_edge=2,
                stratified=False,
                generator=g,
            )
            return result.positions

        torch.autograd.gradcheck(
            func, (vertices,), eps=1e-6, atol=1e-4, raise_exception=True
        )

    def test_gradient_flows_to_vertices(self):
        """Verify that loss.backward() populates vertices.grad."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ],
            requires_grad=True,
        )
        sil = SilhouetteEdges(
            edge_indices=torch.tensor([0]),
            edges=torch.tensor([[0, 1]]),
            front_face=torch.tensor([0]),
            back_face=torch.tensor([-1]),
            batch_size=[1],
        )
        g = torch.Generator().manual_seed(0)
        result = edge_sample(
            vertices, sil, num_samples_per_edge=4, generator=g
        )
        loss = result.positions.sum()
        loss.backward()

        assert vertices.grad is not None
        # Only vertices 0 and 1 participate in edge (0,1)
        assert (vertices.grad[0].abs() > 0).any()
        assert (vertices.grad[1].abs() > 0).any()
        # Vertex 2 is unused
        assert torch.allclose(vertices.grad[2], torch.zeros(3), atol=1e-7)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestEdgeSampleValidation:
    """Input validation edge cases."""

    def test_invalid_vertices_shape_1d(self, two_edge_silhouette):
        """1-D vertices tensor should raise ValueError."""
        with pytest.raises(ValueError, match="vertices must have shape"):
            edge_sample(torch.randn(6), two_edge_silhouette)

    def test_invalid_vertices_shape_wrong_cols(self, two_edge_silhouette):
        """Vertices with 2 columns should raise ValueError."""
        with pytest.raises(ValueError, match="vertices must have shape"):
            edge_sample(torch.randn(3, 2), two_edge_silhouette)

    def test_num_samples_per_edge_zero(
        self, triangle_vertices, two_edge_silhouette
    ):
        """num_samples_per_edge=0 should raise ValueError."""
        with pytest.raises(ValueError, match="num_samples_per_edge"):
            edge_sample(
                triangle_vertices, two_edge_silhouette, num_samples_per_edge=0
            )

    def test_num_samples_per_edge_negative(
        self, triangle_vertices, two_edge_silhouette
    ):
        """Negative num_samples_per_edge should raise ValueError."""
        with pytest.raises(ValueError, match="num_samples_per_edge"):
            edge_sample(
                triangle_vertices, two_edge_silhouette, num_samples_per_edge=-1
            )

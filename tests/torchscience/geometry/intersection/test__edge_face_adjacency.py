"""Tests for edge_face_adjacency."""

import pytest
import torch

from torchscience.geometry.intersection._edge_face_adjacency import (
    edge_face_adjacency,
)
from torchscience.geometry.intersection._edge_sampling_result import (
    EdgeFaceAdjacency,
)


class TestEdgeFaceAdjacencySingleTriangle:
    """Single triangle should produce 3 boundary edges."""

    def test_single_triangle_num_edges(self):
        """A single triangle has exactly 3 edges."""
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        assert isinstance(adj, EdgeFaceAdjacency)
        assert adj.edges.shape == (3, 2)

    def test_single_triangle_all_boundary(self):
        """All edges of a single triangle are boundary edges."""
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        assert adj.is_boundary.all()
        assert adj.is_boundary.sum().item() == 3

    def test_single_triangle_face_indices(self):
        """All edges reference face 0, with face_1 = -1."""
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        assert (adj.face_0 == 0).all()
        assert (adj.face_1 == -1).all()


class TestEdgeFaceAdjacencyTwoTriangles:
    """Two triangles sharing an edge."""

    def test_two_triangles_num_edges(self):
        """Two triangles sharing one edge have 5 unique edges."""
        #  Triangle 0: (0, 1, 2)
        #  Triangle 1: (1, 3, 2)
        #  Shared edge: (1, 2)
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        assert adj.edges.shape == (5, 2)

    def test_two_triangles_boundary_count(self):
        """4 boundary edges and 1 interior edge."""
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        assert adj.is_boundary.sum().item() == 4
        assert (~adj.is_boundary).sum().item() == 1

    def test_shared_edge_references_both_faces(self):
        """The shared edge should reference both face 0 and face 1."""
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        # Find the interior (shared) edge
        interior_mask = ~adj.is_boundary
        assert interior_mask.sum().item() == 1

        interior_idx = interior_mask.nonzero(as_tuple=True)[0][0]
        f0 = adj.face_0[interior_idx].item()
        f1 = adj.face_1[interior_idx].item()

        # Both faces should be referenced
        face_set = {f0, f1}
        assert face_set == {0, 1}

    def test_shared_edge_is_canonical(self):
        """The shared edge (1, 2) should be stored canonically."""
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        interior_mask = ~adj.is_boundary
        interior_idx = interior_mask.nonzero(as_tuple=True)[0][0]
        shared_edge = adj.edges[interior_idx]

        # Canonical: (1, 2) with 1 < 2
        assert shared_edge[0].item() == 1
        assert shared_edge[1].item() == 2


class TestEdgeFaceAdjacencyTetrahedron:
    """Tetrahedron surface (closed manifold) -- 4 faces, 6 edges, no boundary."""

    @pytest.fixture
    def tetrahedron_faces(self):
        """4 triangular faces of a tetrahedron with vertices 0,1,2,3."""
        return torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ],
            dtype=torch.long,
        )

    def test_tetrahedron_num_edges(self, tetrahedron_faces):
        """A tetrahedron has 6 edges."""
        adj = edge_face_adjacency(tetrahedron_faces)
        assert adj.edges.shape == (6, 2)

    def test_tetrahedron_no_boundary(self, tetrahedron_faces):
        """A closed tetrahedron has no boundary edges."""
        adj = edge_face_adjacency(tetrahedron_faces)
        assert not adj.is_boundary.any()

    def test_tetrahedron_all_face_1_valid(self, tetrahedron_faces):
        """All edges should have a valid second face (no -1)."""
        adj = edge_face_adjacency(tetrahedron_faces)
        assert (adj.face_1 >= 0).all()

    def test_tetrahedron_face_indices_in_range(self, tetrahedron_faces):
        """All face indices should be in [0, 3]."""
        adj = edge_face_adjacency(tetrahedron_faces)
        assert (adj.face_0 >= 0).all() and (adj.face_0 < 4).all()
        assert (adj.face_1 >= 0).all() and (adj.face_1 < 4).all()


class TestEdgeFaceAdjacencyCanonicalOrdering:
    """Canonical ordering: edges[:, 0] < edges[:, 1]."""

    def test_canonical_single_triangle(self):
        """All edges in canonical order for a single triangle."""
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        assert (adj.edges[:, 0] < adj.edges[:, 1]).all()

    def test_canonical_reversed_winding(self):
        """Canonical ordering holds even with reversed winding."""
        faces = torch.tensor([[2, 1, 0]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        assert (adj.edges[:, 0] < adj.edges[:, 1]).all()

    def test_canonical_large_indices(self):
        """Canonical ordering holds for large vertex indices."""
        faces = torch.tensor([[100, 200, 300]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        assert (adj.edges[:, 0] < adj.edges[:, 1]).all()
        # Check edge vertex values are correct
        edge_set = {tuple(e.tolist()) for e in adj.edges}
        assert edge_set == {(100, 200), (200, 300), (100, 300)}


class TestEdgeFaceAdjacencyValidation:
    """Input validation tests."""

    def test_reject_1d_input(self):
        """1D tensor should be rejected."""
        faces = torch.tensor([0, 1, 2], dtype=torch.long)

        with pytest.raises(ValueError, match="2D tensor"):
            edge_face_adjacency(faces)

    def test_reject_3d_input(self):
        """3D tensor should be rejected."""
        faces = torch.tensor([[[0, 1, 2]]], dtype=torch.long)

        with pytest.raises(ValueError, match="2D tensor"):
            edge_face_adjacency(faces)

    def test_reject_quads(self):
        """Quad faces (N, 4) should be rejected."""
        faces = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        with pytest.raises(ValueError, match="3 columns"):
            edge_face_adjacency(faces)

    def test_reject_wrong_dtype(self):
        """Non-long dtype should be rejected."""
        faces = torch.tensor([[0, 1, 2]], dtype=torch.float32)

        with pytest.raises(ValueError, match="torch.long"):
            edge_face_adjacency(faces)

    def test_reject_int32_dtype(self):
        """int32 dtype should be rejected (must be int64/long)."""
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        with pytest.raises(ValueError, match="torch.long"):
            edge_face_adjacency(faces)


class TestEdgeFaceAdjacencyNonManifold:
    """Non-manifold mesh detection."""

    def test_non_manifold_edge_raises(self):
        """An edge shared by 3 faces should raise ValueError."""
        # Three faces all sharing edge (0, 1)
        faces = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
            ],
            dtype=torch.long,
        )

        with pytest.raises(ValueError, match="Non-manifold"):
            edge_face_adjacency(faces)


class TestEdgeFaceAdjacencyEdgeCases:
    """Additional edge cases."""

    def test_single_face_edge_vertices(self):
        """Verify the exact edges extracted from a single face."""
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        edge_set = {tuple(e.tolist()) for e in adj.edges}
        expected = {(0, 1), (1, 2), (0, 2)}
        assert edge_set == expected

    def test_strip_of_three_triangles(self):
        """A strip of 3 triangles: 7 unique edges."""
        #  0---1---3---5
        #  | / | / | /
        #  2   4   6
        #  But simpler: a strip sharing sequential edges
        faces = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
                [2, 3, 4],
            ],
            dtype=torch.long,
        )
        adj = edge_face_adjacency(faces)

        # 3 faces * 3 edges = 9 half-edges
        # Shared edges: (1,2) between faces 0 and 1, (2,3) between faces 1 and 2
        # Unique edges: 9 - 2 = 7
        assert adj.edges.shape[0] == 7
        # 2 interior + 5 boundary
        assert (~adj.is_boundary).sum().item() == 2
        assert adj.is_boundary.sum().item() == 5

    def test_output_device_matches_input(self):
        """Output tensors should be on the same device as input."""
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        assert adj.edges.device == faces.device
        assert adj.face_0.device == faces.device
        assert adj.face_1.device == faces.device
        assert adj.is_boundary.device == faces.device

    def test_output_dtypes(self):
        """Check output tensor dtypes."""
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        adj = edge_face_adjacency(faces)

        assert adj.edges.dtype == torch.long
        assert adj.face_0.dtype == torch.long
        assert adj.face_1.dtype == torch.long
        assert adj.is_boundary.dtype == torch.bool

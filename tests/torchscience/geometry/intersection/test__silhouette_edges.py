"""Tests for silhouette_edges."""

import pytest
import torch

from torchscience.geometry.intersection._edge_face_adjacency import (
    edge_face_adjacency,
)
from torchscience.geometry.intersection._edge_sampling_result import (
    SilhouetteEdges,
)
from torchscience.geometry.intersection._silhouette_edges import (
    silhouette_edges,
)


class TestSilhouetteEdgesSingleFrontFacingTriangle:
    """Single front-facing triangle: all 3 boundary edges are silhouette."""

    def test_num_silhouette_edges(self):
        """A single front-facing triangle has 3 silhouette edges."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        # Normal is (0,0,1) via right-hand rule; view along +z is front-facing
        view = torch.tensor([0.0, 0.0, 1.0])

        result = silhouette_edges(vertices, faces, view)

        assert isinstance(result, SilhouetteEdges)
        assert result.edge_indices.shape[0] == 3

    def test_all_boundary_silhouette(self):
        """All silhouette edges should have back_face == -1 (boundary)."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        result = silhouette_edges(vertices, faces, view)

        assert (result.back_face == -1).all()

    def test_front_face_is_zero(self):
        """All silhouette edges should reference face 0 as front face."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        result = silhouette_edges(vertices, faces, view)

        assert (result.front_face == 0).all()


class TestSilhouetteEdgesSingleBackFacingTriangle:
    """Single back-facing triangle: 0 silhouette edges."""

    def test_no_silhouette_edges(self):
        """A single back-facing triangle has no silhouette edges."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        # Normal is (0,0,1); view along -z makes it back-facing
        view = torch.tensor([0.0, 0.0, -1.0])

        result = silhouette_edges(vertices, faces, view)

        assert result.edge_indices.shape[0] == 0

    def test_empty_result_shapes(self):
        """Empty result should have consistent shapes."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, -1.0])

        result = silhouette_edges(vertices, faces, view)

        assert result.edges.shape == (0, 2)
        assert result.front_face.shape == (0,)
        assert result.back_face.shape == (0,)


class TestSilhouetteEdgesTwoCoplanarFrontFacing:
    """Two coplanar front-facing triangles sharing an edge.

    The shared edge is NOT silhouette (both faces front-facing).
    Only 4 boundary edges are silhouette.
    """

    def test_num_silhouette_edges(self):
        """Two coplanar front-facing triangles: 4 boundary silhouette edges."""
        # Two coplanar triangles in the XY plane sharing edge (1,2)
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [0.5, 1.0, 0.0],  # 2
                [1.5, 1.0, 0.0],  # 3
            ]
        )
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        # Both normals point in +z direction; view along +z is front-facing
        view = torch.tensor([0.0, 0.0, 1.0])

        result = silhouette_edges(vertices, faces, view)

        assert result.edge_indices.shape[0] == 4

    def test_shared_edge_excluded(self):
        """The shared edge should NOT be in the silhouette set."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
            ]
        )
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        result = silhouette_edges(vertices, faces, view)

        # Shared edge is (1, 2) in canonical form
        sil_edge_set = {tuple(e.tolist()) for e in result.edges}
        assert (1, 2) not in sil_edge_set

    def test_all_boundary_back_face(self):
        """All silhouette edges should have back_face == -1."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
            ]
        )
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        result = silhouette_edges(vertices, faces, view)

        assert (result.back_face == -1).all()


class TestSilhouetteEdgesFoldedDihedral:
    """Two triangles forming a V-fold (dihedral angle).

    When viewed from the side, the shared edge is silhouette because
    one face is front-facing and the other is back-facing.
    """

    def test_shared_edge_is_silhouette(self):
        """The shared edge of a folded dihedral is a silhouette edge."""
        # Two triangles sharing edge (1,2), folded along that edge
        # Face 0: in XY plane, normal = (0, 0, 1)
        # Face 1: folded back, normal has negative z component
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # 0 - left vertex of face 0
                [1.0, 0.0, 0.0],  # 1 - shared
                [0.5, 1.0, 0.0],  # 2 - shared (top)
                [0.5, 0.0, -1.0],  # 3 - right vertex of face 1 (folded back)
            ]
        )
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        # View from +z direction: face 0 is front, face 1 is back
        view = torch.tensor([0.0, 0.0, 1.0])

        result = silhouette_edges(vertices, faces, view)

        # The shared edge (1, 2) should be in the silhouette set
        sil_edge_set = {tuple(e.tolist()) for e in result.edges}
        assert (1, 2) in sil_edge_set

    def test_shared_edge_front_back_faces(self):
        """The shared silhouette edge should have correct front/back faces."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.0, -1.0],
            ]
        )
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        result = silhouette_edges(vertices, faces, view)

        # Find the shared edge (1, 2) in the results
        for i in range(result.edges.shape[0]):
            edge = tuple(result.edges[i].tolist())
            if edge == (1, 2):
                # Face 0 is front-facing (normal = +z, view = +z)
                assert result.front_face[i].item() == 0
                # Face 1 is back-facing
                assert result.back_face[i].item() == 1
                break
        else:
            pytest.fail("Shared edge (1, 2) not found in silhouette edges")


class TestSilhouetteEdgesClosedTetrahedron:
    """Closed tetrahedron: all silhouette edges have back_face >= 0."""

    @pytest.fixture
    def tetrahedron(self):
        """Regular-ish tetrahedron with consistent winding."""
        vertices = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
            ]
        )
        # Faces with outward-pointing normals
        faces = torch.tensor(
            [
                [0, 1, 2],
                [0, 3, 1],
                [0, 2, 3],
                [1, 3, 2],
            ],
            dtype=torch.long,
        )
        return vertices, faces

    def test_silhouette_edges_have_valid_back_face(self, tetrahedron):
        """All silhouette edges of a closed mesh have back_face >= 0."""
        vertices, faces = tetrahedron
        view = torch.tensor([1.0, 0.0, 0.0])

        result = silhouette_edges(vertices, faces, view)

        # Closed mesh has no boundary, so all silhouette edges are interior
        assert (result.back_face >= 0).all()

    def test_silhouette_count_reasonable(self, tetrahedron):
        """A tetrahedron viewed from any direction has some silhouette edges."""
        vertices, faces = tetrahedron
        view = torch.tensor([1.0, 0.0, 0.0])

        result = silhouette_edges(vertices, faces, view)

        # A tetrahedron has 6 edges total; the silhouette should be
        # a strict subset (at least 1, at most 6)
        assert result.edge_indices.shape[0] > 0
        assert result.edge_indices.shape[0] <= 6

    def test_front_back_faces_differ(self, tetrahedron):
        """Front and back faces should differ for interior silhouette edges."""
        vertices, faces = tetrahedron
        view = torch.tensor([1.0, 0.0, 0.0])

        result = silhouette_edges(vertices, faces, view)

        assert (result.front_face != result.back_face).all()


class TestSilhouetteEdgesPrecomputedAdjacency:
    """Precomputed adjacency gives the same result as auto-computed."""

    def test_precomputed_matches_auto(self):
        """Using precomputed adjacency produces identical results."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
            ]
        )
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        # Auto-computed
        result_auto = silhouette_edges(vertices, faces, view)

        # Precomputed
        adj = edge_face_adjacency(faces)
        result_pre = silhouette_edges(vertices, faces, view, adjacency=adj)

        assert torch.equal(result_auto.edge_indices, result_pre.edge_indices)
        assert torch.equal(result_auto.edges, result_pre.edges)
        assert torch.equal(result_auto.front_face, result_pre.front_face)
        assert torch.equal(result_auto.back_face, result_pre.back_face)


class TestSilhouetteEdgesPerFaceViewDirection:
    """Per-face view direction (num_faces, 3) works."""

    def test_per_face_view_direction(self):
        """Per-face view direction produces correct results."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

        # Same direction for all faces (should match broadcast result)
        view_broadcast = torch.tensor([0.0, 0.0, 1.0])
        view_per_face = torch.tensor([[0.0, 0.0, 1.0]])

        result_broadcast = silhouette_edges(vertices, faces, view_broadcast)
        result_per_face = silhouette_edges(vertices, faces, view_per_face)

        assert (
            result_broadcast.edge_indices.shape
            == result_per_face.edge_indices.shape
        )
        assert torch.equal(
            result_broadcast.edge_indices, result_per_face.edge_indices
        )

    def test_per_face_mixed_directions(self):
        """Per-face view directions can make faces individually front/back."""
        # Two coplanar triangles, but per-face view makes one front and one back
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
            ]
        )
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)

        # Face 0 front-facing (+z view), face 1 back-facing (-z view)
        view_per_face = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

        result = silhouette_edges(vertices, faces, view_per_face)

        # The shared edge (1, 2) should now be silhouette
        # (one face front, one face back)
        sil_edge_set = {tuple(e.tolist()) for e in result.edges}
        assert (1, 2) in sil_edge_set


class TestSilhouetteEdgesValidation:
    """Input validation tests."""

    def test_reject_1d_vertices(self):
        """1D vertices tensor should be rejected."""
        vertices = torch.tensor([0.0, 1.0, 2.0])
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        with pytest.raises(ValueError, match="num_vertices, 3"):
            silhouette_edges(vertices, faces, view)

    def test_reject_wrong_vertices_columns(self):
        """Vertices with wrong number of columns should be rejected."""
        vertices = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        with pytest.raises(ValueError, match="num_vertices, 3"):
            silhouette_edges(vertices, faces, view)

    def test_reject_wrong_faces_shape(self):
        """Faces with wrong shape should be rejected."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([0, 1, 2], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        with pytest.raises(ValueError, match="num_faces, 3"):
            silhouette_edges(vertices, faces, view)

    def test_reject_wrong_faces_dtype(self):
        """Faces with wrong dtype should be rejected."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.float32)
        view = torch.tensor([0.0, 0.0, 1.0])

        with pytest.raises(ValueError, match="torch.long"):
            silhouette_edges(vertices, faces, view)

    def test_reject_wrong_view_direction_1d(self):
        """View direction with wrong size should be rejected."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0])

        with pytest.raises(ValueError, match="shape \\(3,\\)"):
            silhouette_edges(vertices, faces, view)

    def test_reject_wrong_view_direction_2d(self):
        """View direction with wrong 2D shape should be rejected."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        view = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        with pytest.raises(ValueError, match="\\(1, 3\\)"):
            silhouette_edges(vertices, faces, view)

    def test_reject_3d_view_direction(self):
        """3D view direction should be rejected."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        view = torch.tensor([[[0.0, 0.0, 1.0]]])

        with pytest.raises(ValueError, match="1D or 2D"):
            silhouette_edges(vertices, faces, view)


class TestSilhouetteEdgesOutputProperties:
    """Check output tensor shapes, dtypes, and device consistency."""

    def test_output_shapes(self):
        """Output tensors have correct shapes."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        result = silhouette_edges(vertices, faces, view)
        num_sil = result.edge_indices.shape[0]

        assert result.edge_indices.shape == (num_sil,)
        assert result.edges.shape == (num_sil, 2)
        assert result.front_face.shape == (num_sil,)
        assert result.back_face.shape == (num_sil,)

    def test_output_dtypes(self):
        """Output tensors have correct dtypes."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        result = silhouette_edges(vertices, faces, view)

        assert result.edge_indices.dtype == torch.long
        assert result.edges.dtype == torch.long
        assert result.front_face.dtype == torch.long
        assert result.back_face.dtype == torch.long

    def test_output_device_matches_input(self):
        """Output tensors should be on the same device as input."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        result = silhouette_edges(vertices, faces, view)

        assert result.edge_indices.device == faces.device
        assert result.edges.device == faces.device
        assert result.front_face.device == faces.device
        assert result.back_face.device == faces.device

    def test_edge_indices_valid_range(self):
        """edge_indices should be valid indices into the adjacency edges."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
            ]
        )
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        view = torch.tensor([0.0, 0.0, 1.0])

        adj = edge_face_adjacency(faces)
        result = silhouette_edges(vertices, faces, view, adjacency=adj)

        assert (result.edge_indices >= 0).all()
        assert (result.edge_indices < adj.edges.shape[0]).all()

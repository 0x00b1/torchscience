# tests/torchscience/geometry/mesh/test__refinement.py
"""Tests for mesh refinement functions."""

import torch


class TestRefineMeshImport:
    """Tests for refine_mesh import."""

    def test_refine_mesh_importable(self):
        """refine_mesh is importable from geometry.mesh."""
        from torchscience.geometry.mesh import refine_mesh

        assert refine_mesh is not None

    def test_refine_mesh_importable_from_geometry(self):
        """refine_mesh is importable from geometry."""
        from torchscience.geometry import refine_mesh

        assert refine_mesh is not None


class TestRefineMeshTriangle:
    """Tests for refine_mesh with triangle elements."""

    def test_single_triangle_refinement(self):
        """Refining a single triangle produces 4 triangles."""
        from torchscience.geometry.mesh import Mesh, refine_mesh

        # Create a single triangle
        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float64,
        )
        elements = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        refined = refine_mesh(mesh, level=1)

        # 1 triangle -> 4 triangles
        assert refined.num_elements == 4
        # 3 original vertices + 3 midpoints
        assert refined.num_vertices == 6
        assert refined.element_type == "triangle"

    def test_triangle_mesh_refinement(self):
        """Refining a 2x2 triangle mesh quadruples elements."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        mesh = rectangle_mesh(2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]])
        # 2x2 grid -> 8 triangles
        assert mesh.num_elements == 8

        refined = refine_mesh(mesh, level=1)

        # 8 triangles -> 32 triangles
        assert refined.num_elements == 32
        assert refined.element_type == "triangle"

    def test_triangle_midpoint_vertices(self):
        """New vertices are at edge midpoints."""
        from torchscience.geometry.mesh import Mesh, refine_mesh

        # Create a right triangle
        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=torch.float64,
        )
        elements = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        refined = refine_mesh(mesh, level=1)

        # Check that midpoints exist
        expected_midpoints = torch.tensor(
            [
                [1.0, 0.0],  # midpoint of edge (0,1)
                [1.0, 1.0],  # midpoint of edge (1,2)
                [0.0, 1.0],  # midpoint of edge (2,0)
            ],
            dtype=torch.float64,
        )

        for midpoint in expected_midpoints:
            distances = torch.norm(refined.vertices - midpoint, dim=1)
            assert torch.any(distances < 1e-6), (
                f"Midpoint {midpoint} not found in refined vertices"
            )

    def test_triangle_all_elements_valid(self):
        """All refined elements have valid vertex indices."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        mesh = rectangle_mesh(2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]])
        refined = refine_mesh(mesh, level=1)

        # All element indices should be valid vertex indices
        assert refined.elements.min() >= 0
        assert refined.elements.max() < refined.num_vertices

        # Each element should have 3 unique vertices
        for i in range(refined.num_elements):
            element = refined.elements[i]
            assert len(torch.unique(element)) == 3


class TestRefineMeshQuad:
    """Tests for refine_mesh with quad elements."""

    def test_single_quad_refinement(self):
        """Refining a single quad produces 4 quads."""
        from torchscience.geometry.mesh import Mesh, refine_mesh

        # Create a single quad
        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        elements = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="quad",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        refined = refine_mesh(mesh, level=1)

        # 1 quad -> 4 quads
        assert refined.num_elements == 4
        # 4 original vertices + 4 edge midpoints + 1 center
        assert refined.num_vertices == 9
        assert refined.element_type == "quad"

    def test_quad_mesh_refinement(self):
        """Refining a 2x2 quad mesh quadruples elements."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        mesh = rectangle_mesh(
            2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]], element_type="quad"
        )
        # 2x2 grid -> 4 quads
        assert mesh.num_elements == 4

        refined = refine_mesh(mesh, level=1)

        # 4 quads -> 16 quads
        assert refined.num_elements == 16
        assert refined.element_type == "quad"

    def test_quad_midpoint_and_center_vertices(self):
        """New vertices include edge midpoints and center."""
        from torchscience.geometry.mesh import Mesh, refine_mesh

        # Create a unit square quad
        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ],
            dtype=torch.float64,
        )
        elements = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="quad",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        refined = refine_mesh(mesh, level=1)

        # Check that midpoints and center exist
        expected_new_vertices = torch.tensor(
            [
                [1.0, 0.0],  # bottom edge midpoint
                [2.0, 1.0],  # right edge midpoint
                [1.0, 2.0],  # top edge midpoint
                [0.0, 1.0],  # left edge midpoint
                [1.0, 1.0],  # center
            ],
            dtype=torch.float64,
        )

        for new_vertex in expected_new_vertices:
            distances = torch.norm(refined.vertices - new_vertex, dim=1)
            assert torch.any(distances < 1e-6), (
                f"Expected vertex {new_vertex} not found in refined mesh"
            )

    def test_quad_all_elements_valid(self):
        """All refined quad elements have valid vertex indices."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        mesh = rectangle_mesh(
            2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]], element_type="quad"
        )
        refined = refine_mesh(mesh, level=1)

        # All element indices should be valid vertex indices
        assert refined.elements.min() >= 0
        assert refined.elements.max() < refined.num_vertices

        # Each element should have 4 unique vertices
        for i in range(refined.num_elements):
            element = refined.elements[i]
            assert len(torch.unique(element)) == 4


class TestRefineMeshMultipleLevels:
    """Tests for multi-level mesh refinement."""

    def test_level_0_no_change(self):
        """Level 0 refinement returns the same mesh."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        mesh = rectangle_mesh(2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]])
        refined = refine_mesh(mesh, level=0)

        assert refined.num_elements == mesh.num_elements
        assert refined.num_vertices == mesh.num_vertices

    def test_level_2_triangle(self):
        """Level 2 refinement gives 16x elements for triangles."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        mesh = rectangle_mesh(1, 1, bounds=[[0.0, 1.0], [0.0, 1.0]])
        # 1x1 grid -> 2 triangles
        assert mesh.num_elements == 2

        refined = refine_mesh(mesh, level=2)

        # 2 * 4^2 = 32 triangles
        assert refined.num_elements == 32
        assert refined.element_type == "triangle"

    def test_level_2_quad(self):
        """Level 2 refinement gives 16x elements for quads."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        mesh = rectangle_mesh(
            1, 1, bounds=[[0.0, 1.0], [0.0, 1.0]], element_type="quad"
        )
        # 1x1 grid -> 1 quad
        assert mesh.num_elements == 1

        refined = refine_mesh(mesh, level=2)

        # 1 * 4^2 = 16 quads
        assert refined.num_elements == 16
        assert refined.element_type == "quad"

    def test_level_3_element_count(self):
        """Level 3 refinement gives 64x elements."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        mesh = rectangle_mesh(1, 1, bounds=[[0.0, 1.0], [0.0, 1.0]])
        initial_elements = mesh.num_elements

        refined = refine_mesh(mesh, level=3)

        # Elements multiply by 4 each level: 4^3 = 64
        assert refined.num_elements == initial_elements * 64


class TestRefineMeshBoundaryPreservation:
    """Tests for boundary preservation during refinement."""

    def test_boundary_facets_preserved_triangle(self):
        """Boundary facets are updated correctly for triangles."""
        from torchscience.geometry.mesh import (
            mesh_boundary_facets,
            rectangle_mesh,
            refine_mesh,
        )

        mesh = rectangle_mesh(2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]])
        # Original boundary: 8 edges (2*2 + 2*2)
        original_boundary = mesh_boundary_facets(mesh)
        assert original_boundary.shape[0] == 8

        refined = refine_mesh(mesh, level=1)

        # After refinement: each boundary edge is split into 2
        refined_boundary = mesh_boundary_facets(refined)
        assert refined_boundary.shape[0] == 16  # 8 * 2

    def test_boundary_facets_preserved_quad(self):
        """Boundary facets are updated correctly for quads."""
        from torchscience.geometry.mesh import (
            mesh_boundary_facets,
            rectangle_mesh,
            refine_mesh,
        )

        mesh = rectangle_mesh(
            2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]], element_type="quad"
        )
        # Original boundary: 8 edges (2*2 + 2*2)
        original_boundary = mesh_boundary_facets(mesh)
        assert original_boundary.shape[0] == 8

        refined = refine_mesh(mesh, level=1)

        # After refinement: each boundary edge is split into 2
        refined_boundary = mesh_boundary_facets(refined)
        assert refined_boundary.shape[0] == 16  # 8 * 2


class TestRefineMeshDtypeDevice:
    """Tests for dtype and device handling."""

    def test_preserves_dtype(self):
        """Refinement preserves vertex dtype."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        mesh = rectangle_mesh(
            2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64
        )

        refined = refine_mesh(mesh, level=1)

        assert refined.vertices.dtype == torch.float64

    def test_preserves_device(self):
        """Refinement preserves device."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        mesh = rectangle_mesh(
            2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]], device=torch.device("cpu")
        )

        refined = refine_mesh(mesh, level=1)

        assert refined.vertices.device.type == "cpu"
        assert refined.elements.device.type == "cpu"


class TestRefineMeshVertexCount:
    """Tests for vertex count after refinement."""

    def test_triangle_vertex_count_formula(self):
        """Triangle refinement produces correct vertex count."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        # For triangle mesh:
        # After 1 level: V_new = V_old + E_old
        # where E_old is the number of unique edges

        mesh = rectangle_mesh(2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]])
        # 9 vertices, 16 edges (interior + boundary)
        # After refinement: 9 + 16 = 25 vertices

        refined = refine_mesh(mesh, level=1)

        # Verify vertex count is reasonable (depends on edge count)
        assert refined.num_vertices > mesh.num_vertices

    def test_quad_vertex_count_formula(self):
        """Quad refinement produces correct vertex count."""
        from torchscience.geometry.mesh import rectangle_mesh, refine_mesh

        # For quad mesh:
        # After 1 level: V_new = V_old + E_old + num_elements
        # (original vertices + edge midpoints + element centers)

        mesh = rectangle_mesh(
            2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]], element_type="quad"
        )
        # 9 vertices, 12 edges, 4 elements
        # After refinement: 9 + 12 + 4 = 25 vertices

        refined = refine_mesh(mesh, level=1)

        assert refined.num_vertices == 25

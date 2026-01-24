"""Tests for mesh boundary detection."""

import torch

from torchscience.geometry.mesh import (
    mesh_boundary_facets,
    mesh_boundary_vertices,
    rectangle_mesh,
)


class TestMeshBoundary:
    def test_boundary_facets_triangle(self):
        """Test boundary facet detection for triangle mesh."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        boundary = mesh_boundary_facets(mesh)

        # 2x2 mesh has 8 boundary edges
        assert boundary.shape[0] == 8
        assert boundary.shape[1] == 2  # 2 vertices per edge

    def test_boundary_vertices_triangle(self):
        """Test boundary vertex detection."""
        mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
        boundary_verts = mesh_boundary_vertices(mesh)

        # All vertices on perimeter: 4 corners + 4 edge midpoints = 8
        assert boundary_verts.shape[0] == 8

    def test_boundary_vertices_are_on_boundary(self):
        """Test that detected boundary vertices are actually on the boundary."""
        mesh = rectangle_mesh(3, 3, bounds=[[0, 1], [0, 1]])
        boundary_verts = mesh_boundary_vertices(mesh)

        # All boundary vertices should be on edge (x=0, x=1, y=0, or y=1)
        coords = mesh.vertices[boundary_verts]
        on_x0 = torch.isclose(
            coords[:, 0], torch.tensor(0.0, dtype=coords.dtype)
        )
        on_x1 = torch.isclose(
            coords[:, 0], torch.tensor(1.0, dtype=coords.dtype)
        )
        on_y0 = torch.isclose(
            coords[:, 1], torch.tensor(0.0, dtype=coords.dtype)
        )
        on_y1 = torch.isclose(
            coords[:, 1], torch.tensor(1.0, dtype=coords.dtype)
        )

        assert (on_x0 | on_x1 | on_y0 | on_y1).all()

    def test_boundary_facets_quad(self):
        """Test boundary facet detection for quad mesh."""
        mesh = rectangle_mesh(
            2, 2, bounds=[[0, 1], [0, 1]], element_type="quad"
        )
        boundary = mesh_boundary_facets(mesh)

        # 2x2 quad mesh has 8 boundary edges
        assert boundary.shape[0] == 8
        assert boundary.shape[1] == 2  # 2 vertices per edge

    def test_boundary_facets_3d(self):
        """Test boundary facet detection for 3D tetrahedron mesh."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            2,
            2,
            2,
            bounds=[[0, 1], [0, 1], [0, 1]],
            element_type="tetrahedron",
        )
        boundary = mesh_boundary_facets(mesh)

        # Boundary facets are triangles (3 vertices each)
        assert boundary.shape[1] == 3

        # A 2x2x2 box has 6 faces, each divided into triangles
        # Each face is 2x2 quads = 4 quads per face = 8 triangles per face
        # 6 faces * 8 triangles = 48 boundary triangles
        assert boundary.shape[0] == 48

    def test_boundary_vertices_3d(self):
        """Test boundary vertex detection for 3D mesh."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            2,
            2,
            2,
            bounds=[[0, 1], [0, 1], [0, 1]],
            element_type="tetrahedron",
        )
        boundary_verts = mesh_boundary_vertices(mesh)

        # All 27 vertices of a 3x3x3 grid are on the boundary for this mesh
        # (interior vertices only exist for larger meshes)
        # Actually for 2x2x2 elements, we have (2+1)^3 = 27 vertices
        # Boundary vertices are those on any face: all 27 for this small mesh
        assert boundary_verts.shape[0] == 26  # All except center

    def test_invalid_element_type(self):
        """Test error handling for invalid element type."""
        import pytest

        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        elements = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="invalid_type",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        with pytest.raises(ValueError, match="Unknown element type"):
            mesh_boundary_facets(mesh)

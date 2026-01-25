# tests/torchscience/geometry/mesh/test__rectangle_mesh.py
"""Tests for rectangle_mesh mesh generator."""

import torch


class TestRectangleMeshImport:
    """Tests for rectangle_mesh import."""

    def test_rectangle_mesh_importable(self):
        """rectangle_mesh is importable from geometry.mesh."""
        from torchscience.geometry.mesh import rectangle_mesh

        assert rectangle_mesh is not None

    def test_rectangle_mesh_importable_from_geometry(self):
        """rectangle_mesh is importable from geometry."""
        from torchscience.geometry import rectangle_mesh

        assert rectangle_mesh is not None


class TestRectangleMeshTriangle:
    """Tests for rectangle_mesh with triangle elements."""

    def test_triangle_mesh_basic(self):
        """Create a basic 2x2 triangle mesh on unit square."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
        )

        # 2x2 grid of quads -> 3x3 vertices
        assert mesh.num_vertices == 9
        # Each quad is split into 2 triangles -> 2*2*2 = 8 triangles
        assert mesh.num_elements == 8
        assert mesh.element_type == "triangle"
        assert mesh.nodes_per_element == 3
        assert mesh.dim == 2

    def test_triangle_mesh_vertices_shape(self):
        """Triangle mesh has correct vertex shape."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=3,
            ny=4,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
        )

        # (nx+1) * (ny+1) vertices, each with 2 coordinates
        assert mesh.vertices.shape == (20, 2)  # 4 * 5 = 20

    def test_triangle_mesh_elements_shape(self):
        """Triangle mesh has correct element shape."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=3,
            ny=4,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
        )

        # nx * ny quads, each split into 2 triangles
        assert mesh.elements.shape == (24, 3)  # 3 * 4 * 2 = 24

    def test_triangle_mesh_vertex_positions(self):
        """Triangle mesh vertices are at correct positions."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
        )

        # Check corner vertices exist
        vertices = mesh.vertices

        # Check that corners are present (order may vary)
        corners = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )

        for corner in corners:
            # Check if this corner exists in vertices
            distances = torch.norm(vertices - corner, dim=1)
            assert torch.any(distances < 1e-6), (
                f"Corner {corner} not found in vertices"
            )

    def test_triangle_mesh_custom_bounds(self):
        """Triangle mesh respects custom bounds."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[-1.0, 2.0], [0.5, 1.5]],
            element_type="triangle",
        )

        # Check bounds
        assert torch.allclose(
            mesh.vertices[:, 0].min(), torch.tensor(-1.0, dtype=torch.float64)
        )
        assert torch.allclose(
            mesh.vertices[:, 0].max(), torch.tensor(2.0, dtype=torch.float64)
        )
        assert torch.allclose(
            mesh.vertices[:, 1].min(), torch.tensor(0.5, dtype=torch.float64)
        )
        assert torch.allclose(
            mesh.vertices[:, 1].max(), torch.tensor(1.5, dtype=torch.float64)
        )

    def test_triangle_mesh_element_connectivity(self):
        """Triangle mesh has valid element connectivity."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
        )

        # All element indices should be valid vertex indices
        assert mesh.elements.min() >= 0
        assert mesh.elements.max() < mesh.num_vertices

        # Each element should have 3 unique vertices
        for i in range(mesh.num_elements):
            element = mesh.elements[i]
            assert len(torch.unique(element)) == 3


class TestRectangleMeshQuad:
    """Tests for rectangle_mesh with quad elements."""

    def test_quad_mesh_basic(self):
        """Create a basic 2x2 quad mesh on unit square."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="quad",
        )

        # 2x2 grid of quads -> 3x3 vertices
        assert mesh.num_vertices == 9
        # 2x2 = 4 quads
        assert mesh.num_elements == 4
        assert mesh.element_type == "quad"
        assert mesh.nodes_per_element == 4
        assert mesh.dim == 2

    def test_quad_mesh_vertices_shape(self):
        """Quad mesh has correct vertex shape."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=3,
            ny=4,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="quad",
        )

        # (nx+1) * (ny+1) vertices, each with 2 coordinates
        assert mesh.vertices.shape == (20, 2)  # 4 * 5 = 20

    def test_quad_mesh_elements_shape(self):
        """Quad mesh has correct element shape."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=3,
            ny=4,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="quad",
        )

        # nx * ny quads
        assert mesh.elements.shape == (12, 4)  # 3 * 4 = 12

    def test_quad_mesh_element_connectivity(self):
        """Quad mesh has valid element connectivity."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="quad",
        )

        # All element indices should be valid vertex indices
        assert mesh.elements.min() >= 0
        assert mesh.elements.max() < mesh.num_vertices

        # Each element should have 4 unique vertices
        for i in range(mesh.num_elements):
            element = mesh.elements[i]
            assert len(torch.unique(element)) == 4


class TestRectangleMeshDtype:
    """Tests for rectangle_mesh dtype handling."""

    def test_default_dtype_float64(self):
        """Default dtype is float64."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
        )

        assert mesh.vertices.dtype == torch.float64

    def test_custom_dtype_float64(self):
        """Custom dtype float64 is respected."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
            dtype=torch.float64,
        )

        assert mesh.vertices.dtype == torch.float64

    def test_elements_dtype_int64(self):
        """Element tensor has int64 dtype."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
        )

        assert mesh.elements.dtype == torch.int64


class TestRectangleMeshDevice:
    """Tests for rectangle_mesh device handling."""

    def test_default_device_cpu(self):
        """Default device is CPU."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
        )

        assert mesh.vertices.device.type == "cpu"
        assert mesh.elements.device.type == "cpu"

    def test_explicit_device_cpu(self):
        """Explicit CPU device is respected."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
            device=torch.device("cpu"),
        )

        assert mesh.vertices.device.type == "cpu"


class TestRectangleMeshDifferentiability:
    """Tests for rectangle_mesh differentiability w.r.t. bounds."""

    def test_vertices_differentiable_wrt_bounds(self):
        """Vertices are differentiable with respect to bounds."""
        from torchscience.geometry.mesh import rectangle_mesh

        bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True)

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=bounds,
            element_type="triangle",
        )

        # Compute a simple loss (sum of vertex coordinates)
        loss = mesh.vertices.sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert bounds.grad is not None
        assert torch.any(bounds.grad != 0)

    def test_gradient_shape_optimization(self):
        """Gradients flow correctly for shape optimization."""
        from torchscience.geometry.mesh import rectangle_mesh

        bounds = torch.tensor([[0.0, 2.0], [0.0, 1.0]], requires_grad=True)

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=bounds,
            element_type="triangle",
        )

        # Loss: mean x-coordinate (depends on x bounds)
        loss = mesh.vertices[:, 0].mean()
        loss.backward()

        # Gradient w.r.t x_min and x_max should exist
        assert bounds.grad[0, 0] != 0  # gradient w.r.t x_min
        assert bounds.grad[0, 1] != 0  # gradient w.r.t x_max


class TestRectangleMeshBoundary:
    """Tests for rectangle_mesh boundary facet generation."""

    def test_boundary_facets_exist(self):
        """Boundary facets are generated."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
        )

        assert mesh.boundary_facets is not None

    def test_boundary_facets_shape_triangle(self):
        """Triangle mesh has correct number of boundary facets."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="triangle",
        )

        # 2*nx + 2*ny boundary edges
        # 2*2 + 2*2 = 8 boundary edges
        assert mesh.boundary_facets.shape == (8, 2)

    def test_boundary_facets_shape_quad(self):
        """Quad mesh has correct number of boundary facets."""
        from torchscience.geometry.mesh import rectangle_mesh

        mesh = rectangle_mesh(
            nx=3,
            ny=4,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            element_type="quad",
        )

        # 2*nx + 2*ny boundary edges
        # 2*3 + 2*4 = 14 boundary edges
        assert mesh.boundary_facets.shape == (14, 2)


class TestRectangleMeshValidation:
    """Tests for rectangle_mesh input validation."""

    def test_invalid_element_type(self):
        """Invalid element type raises error."""
        import pytest

        from torchscience.geometry.mesh import rectangle_mesh

        with pytest.raises(ValueError, match="element_type"):
            rectangle_mesh(
                nx=2,
                ny=2,
                bounds=[[0.0, 1.0], [0.0, 1.0]],
                element_type="tetrahedron",  # 3D element type invalid for 2D mesh
            )

    def test_invalid_nx(self):
        """nx < 1 raises error."""
        import pytest

        from torchscience.geometry.mesh import rectangle_mesh

        with pytest.raises(ValueError, match="nx"):
            rectangle_mesh(
                nx=0,
                ny=2,
                bounds=[[0.0, 1.0], [0.0, 1.0]],
                element_type="triangle",
            )

    def test_invalid_ny(self):
        """ny < 1 raises error."""
        import pytest

        from torchscience.geometry.mesh import rectangle_mesh

        with pytest.raises(ValueError, match="ny"):
            rectangle_mesh(
                nx=2,
                ny=0,
                bounds=[[0.0, 1.0], [0.0, 1.0]],
                element_type="triangle",
            )

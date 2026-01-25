# tests/torchscience/geometry/mesh/test__box_mesh.py
"""Tests for box_mesh mesh generator."""

import torch


class TestBoxMeshImport:
    """Tests for box_mesh import."""

    def test_box_mesh_importable(self):
        """box_mesh is importable from geometry.mesh."""
        from torchscience.geometry.mesh import box_mesh

        assert box_mesh is not None

    def test_box_mesh_importable_from_geometry(self):
        """box_mesh is importable from geometry."""
        from torchscience.geometry import box_mesh

        assert box_mesh is not None


class TestBoxMeshTetrahedron:
    """Tests for box_mesh with tetrahedron elements."""

    def test_tetrahedron_mesh_basic(self):
        """Create a basic 2x2x2 tetrahedron mesh on unit cube."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
        )

        # 2x2x2 grid of cubes -> 3x3x3 vertices = 27
        assert mesh.num_vertices == 27
        # Each cube is split into 6 tetrahedra -> 2*2*2*6 = 48 tetrahedra
        assert mesh.num_elements == 48
        assert mesh.element_type == "tetrahedron"
        assert mesh.nodes_per_element == 4
        assert mesh.dim == 3

    def test_tetrahedron_mesh_vertices_shape(self):
        """Tetrahedron mesh has correct vertex shape."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=3,
            ny=4,
            nz=5,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
        )

        # (nx+1) * (ny+1) * (nz+1) vertices, each with 3 coordinates
        assert mesh.vertices.shape == (120, 3)  # 4 * 5 * 6 = 120

    def test_tetrahedron_mesh_elements_shape(self):
        """Tetrahedron mesh has correct element shape."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=3,
            ny=4,
            nz=5,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
        )

        # nx * ny * nz cubes, each split into 6 tetrahedra
        assert mesh.elements.shape == (360, 4)  # 3 * 4 * 5 * 6 = 360

    def test_tetrahedron_mesh_vertex_positions(self):
        """Tetrahedron mesh vertices are at correct positions."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
        )

        # Check corner vertices exist
        vertices = mesh.vertices

        # Check that corners are present
        corners = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        for corner in corners:
            # Check if this corner exists in vertices
            distances = torch.norm(vertices - corner, dim=1)
            assert torch.any(distances < 1e-6), (
                f"Corner {corner} not found in vertices"
            )

    def test_tetrahedron_mesh_custom_bounds(self):
        """Tetrahedron mesh respects custom bounds."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[-1.0, 2.0], [0.5, 1.5], [-0.5, 0.5]],
            element_type="tetrahedron",
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
        assert torch.allclose(
            mesh.vertices[:, 2].min(), torch.tensor(-0.5, dtype=torch.float64)
        )
        assert torch.allclose(
            mesh.vertices[:, 2].max(), torch.tensor(0.5, dtype=torch.float64)
        )

    def test_tetrahedron_mesh_element_connectivity(self):
        """Tetrahedron mesh has valid element connectivity."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
        )

        # All element indices should be valid vertex indices
        assert mesh.elements.min() >= 0
        assert mesh.elements.max() < mesh.num_vertices

        # Each element should have 4 unique vertices
        for i in range(mesh.num_elements):
            element = mesh.elements[i]
            assert len(torch.unique(element)) == 4

    def test_tetrahedron_positive_volume(self):
        """All tetrahedra have positive volume (consistent orientation)."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
        )

        for i in range(mesh.num_elements):
            elem = mesh.elements[i]
            v0 = mesh.vertices[elem[0]]
            v1 = mesh.vertices[elem[1]]
            v2 = mesh.vertices[elem[2]]
            v3 = mesh.vertices[elem[3]]

            # Volume = (1/6) * det([v1-v0, v2-v0, v3-v0])
            mat = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=0)
            det = torch.det(mat)
            volume = det / 6.0

            assert volume > 0, (
                f"Tetrahedron {i} has non-positive volume: {volume}"
            )


class TestBoxMeshHexahedron:
    """Tests for box_mesh with hexahedron elements."""

    def test_hexahedron_mesh_basic(self):
        """Create a basic 2x2x2 hexahedron mesh on unit cube."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="hexahedron",
        )

        # 2x2x2 grid of cubes -> 3x3x3 vertices = 27
        assert mesh.num_vertices == 27
        # 2x2x2 = 8 hexahedra
        assert mesh.num_elements == 8
        assert mesh.element_type == "hexahedron"
        assert mesh.nodes_per_element == 8
        assert mesh.dim == 3

    def test_hexahedron_mesh_vertices_shape(self):
        """Hexahedron mesh has correct vertex shape."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=3,
            ny=4,
            nz=5,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="hexahedron",
        )

        # (nx+1) * (ny+1) * (nz+1) vertices, each with 3 coordinates
        assert mesh.vertices.shape == (120, 3)  # 4 * 5 * 6 = 120

    def test_hexahedron_mesh_elements_shape(self):
        """Hexahedron mesh has correct element shape."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=3,
            ny=4,
            nz=5,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="hexahedron",
        )

        # nx * ny * nz hexahedra
        assert mesh.elements.shape == (60, 8)  # 3 * 4 * 5 = 60

    def test_hexahedron_mesh_element_connectivity(self):
        """Hexahedron mesh has valid element connectivity."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="hexahedron",
        )

        # All element indices should be valid vertex indices
        assert mesh.elements.min() >= 0
        assert mesh.elements.max() < mesh.num_vertices

        # Each element should have 8 unique vertices
        for i in range(mesh.num_elements):
            element = mesh.elements[i]
            assert len(torch.unique(element)) == 8


class TestBoxMeshDtype:
    """Tests for box_mesh dtype handling."""

    def test_default_dtype_float64(self):
        """Default dtype is float64."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
        )

        assert mesh.vertices.dtype == torch.float64

    def test_custom_dtype_float32(self):
        """Custom dtype float32 is respected."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
            dtype=torch.float32,
        )

        assert mesh.vertices.dtype == torch.float32

    def test_elements_dtype_int64(self):
        """Element tensor has int64 dtype."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
        )

        assert mesh.elements.dtype == torch.int64


class TestBoxMeshDevice:
    """Tests for box_mesh device handling."""

    def test_default_device_cpu(self):
        """Default device is CPU."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
        )

        assert mesh.vertices.device.type == "cpu"
        assert mesh.elements.device.type == "cpu"

    def test_explicit_device_cpu(self):
        """Explicit CPU device is respected."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
            device=torch.device("cpu"),
        )

        assert mesh.vertices.device.type == "cpu"


class TestBoxMeshDifferentiability:
    """Tests for box_mesh differentiability w.r.t. bounds."""

    def test_vertices_differentiable_wrt_bounds(self):
        """Vertices are differentiable with respect to bounds."""
        from torchscience.geometry.mesh import box_mesh

        bounds = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True
        )

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=bounds,
            element_type="tetrahedron",
        )

        # Compute a simple loss (sum of vertex coordinates)
        loss = mesh.vertices.sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert bounds.grad is not None
        assert torch.any(bounds.grad != 0)

    def test_gradient_shape_optimization(self):
        """Gradients flow correctly for shape optimization."""
        from torchscience.geometry.mesh import box_mesh

        bounds = torch.tensor(
            [[0.0, 2.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True
        )

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=bounds,
            element_type="tetrahedron",
        )

        # Loss: mean x-coordinate (depends on x bounds)
        loss = mesh.vertices[:, 0].mean()
        loss.backward()

        # Gradient w.r.t x_min and x_max should exist
        assert bounds.grad[0, 0] != 0  # gradient w.r.t x_min
        assert bounds.grad[0, 1] != 0  # gradient w.r.t x_max


class TestBoxMeshBoundary:
    """Tests for box_mesh boundary facet generation."""

    def test_boundary_facets_exist(self):
        """Boundary facets are generated."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
        )

        assert mesh.boundary_facets is not None

    def test_boundary_facets_shape_tetrahedron(self):
        """Tetrahedron mesh has triangular boundary facets."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="tetrahedron",
        )

        # Boundary facets for tetrahedra are triangles (3 nodes)
        assert mesh.boundary_facets.shape[1] == 3

        # 6 faces, each with nx*ny*2 triangles (since each quad is split)
        # = 6 * 2 * 2 * 2 = 48 boundary triangles
        expected_facets = (
            2 * (2 * 2 + 2 * 2 + 2 * 2) * 2
        )  # 2 triangles per quad
        assert mesh.boundary_facets.shape[0] == expected_facets

    def test_boundary_facets_shape_hexahedron(self):
        """Hexahedron mesh has quadrilateral boundary facets."""
        from torchscience.geometry.mesh import box_mesh

        mesh = box_mesh(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            element_type="hexahedron",
        )

        # Boundary facets for hexahedra are quads (4 nodes)
        assert mesh.boundary_facets.shape[1] == 4

        # 6 faces: 2 faces of nx*ny + 2 faces of ny*nz + 2 faces of nx*nz
        # = 2*(2*2) + 2*(2*2) + 2*(2*2) = 24 boundary quads
        expected_facets = 2 * (2 * 2 + 2 * 2 + 2 * 2)
        assert mesh.boundary_facets.shape[0] == expected_facets


class TestBoxMeshValidation:
    """Tests for box_mesh input validation."""

    def test_invalid_element_type(self):
        """Invalid element type raises error."""
        import pytest

        from torchscience.geometry.mesh import box_mesh

        with pytest.raises(ValueError, match="element_type"):
            box_mesh(
                nx=2,
                ny=2,
                nz=2,
                bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                element_type="triangle",  # 2D element type invalid for 3D mesh
            )

    def test_invalid_nx(self):
        """nx < 1 raises error."""
        import pytest

        from torchscience.geometry.mesh import box_mesh

        with pytest.raises(ValueError, match="nx"):
            box_mesh(
                nx=0,
                ny=2,
                nz=2,
                bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                element_type="tetrahedron",
            )

    def test_invalid_ny(self):
        """ny < 1 raises error."""
        import pytest

        from torchscience.geometry.mesh import box_mesh

        with pytest.raises(ValueError, match="ny"):
            box_mesh(
                nx=2,
                ny=0,
                nz=2,
                bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                element_type="tetrahedron",
            )

    def test_invalid_nz(self):
        """nz < 1 raises error."""
        import pytest

        from torchscience.geometry.mesh import box_mesh

        with pytest.raises(ValueError, match="nz"):
            box_mesh(
                nx=2,
                ny=2,
                nz=0,
                bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                element_type="tetrahedron",
            )

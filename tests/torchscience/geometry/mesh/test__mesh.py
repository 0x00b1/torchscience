# tests/torchscience/geometry/mesh/test__mesh.py
"""Tests for Mesh tensorclass."""

import torch


class TestMeshImport:
    """Tests for Mesh import."""

    def test_mesh_importable(self):
        """Mesh class is importable from geometry.mesh."""
        from torchscience.geometry.mesh import Mesh

        assert Mesh is not None

    def test_mesh_importable_from_geometry(self):
        """Mesh class is importable from geometry."""
        from torchscience.geometry import Mesh

        assert Mesh is not None


class TestMeshCreation:
    """Tests for Mesh tensorclass creation."""

    def test_line_mesh_creation(self):
        """Create a simple line mesh."""
        from torchscience.geometry.mesh import Mesh

        # Three line elements forming a polyline
        vertices = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        elements = torch.tensor([[0, 1], [1, 2], [2, 3]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="line",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.vertices.shape == (4, 1)
        assert mesh.elements.shape == (3, 2)
        assert mesh.element_type == "line"

    def test_triangle_mesh_creation(self):
        """Create a simple triangle mesh."""
        from torchscience.geometry.mesh import Mesh

        # Single triangle
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        elements = torch.tensor([[0, 1, 2]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.vertices.shape == (3, 2)
        assert mesh.elements.shape == (1, 3)
        assert mesh.element_type == "triangle"

    def test_quad_mesh_creation(self):
        """Create a simple quad mesh."""
        from torchscience.geometry.mesh import Mesh

        # Single quad
        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        elements = torch.tensor([[0, 1, 2, 3]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="quad",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.vertices.shape == (4, 2)
        assert mesh.elements.shape == (1, 4)
        assert mesh.element_type == "quad"

    def test_tetrahedron_mesh_creation(self):
        """Create a simple tetrahedron mesh."""
        from torchscience.geometry.mesh import Mesh

        # Single tetrahedron
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ]
        )
        elements = torch.tensor([[0, 1, 2, 3]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="tetrahedron",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.vertices.shape == (4, 3)
        assert mesh.elements.shape == (1, 4)
        assert mesh.element_type == "tetrahedron"

    def test_hexahedron_mesh_creation(self):
        """Create a simple hexahedron mesh."""
        from torchscience.geometry.mesh import Mesh

        # Single hexahedron (cube)
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        elements = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="hexahedron",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.vertices.shape == (8, 3)
        assert mesh.elements.shape == (1, 8)
        assert mesh.element_type == "hexahedron"


class TestMeshProperties:
    """Tests for Mesh computed properties."""

    def test_dim_2d(self):
        """dim property returns 2 for 2D mesh."""
        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        elements = torch.tensor([[0, 1, 2]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.dim == 2

    def test_dim_3d(self):
        """dim property returns 3 for 3D mesh."""
        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ]
        )
        elements = torch.tensor([[0, 1, 2, 3]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="tetrahedron",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.dim == 3

    def test_num_vertices(self):
        """num_vertices property returns correct count."""
        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
        )
        elements = torch.tensor([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.num_vertices == 5

    def test_num_elements(self):
        """num_elements property returns correct count."""
        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
        )
        elements = torch.tensor([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.num_elements == 4

    def test_nodes_per_element_triangle(self):
        """nodes_per_element returns 3 for triangle."""
        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        elements = torch.tensor([[0, 1, 2]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.nodes_per_element == 3

    def test_nodes_per_element_quad(self):
        """nodes_per_element returns 4 for quad."""
        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        elements = torch.tensor([[0, 1, 2, 3]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="quad",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.nodes_per_element == 4

    def test_nodes_per_element_tetrahedron(self):
        """nodes_per_element returns 4 for tetrahedron."""
        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ]
        )
        elements = torch.tensor([[0, 1, 2, 3]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="tetrahedron",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.nodes_per_element == 4

    def test_nodes_per_element_hexahedron(self):
        """nodes_per_element returns 8 for hexahedron."""
        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        elements = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="hexahedron",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.nodes_per_element == 8


class TestMeshBoundary:
    """Tests for Mesh with boundary information."""

    def test_mesh_with_boundary_facets(self):
        """Create mesh with boundary facets."""
        from torchscience.geometry.mesh import Mesh

        # Two triangles sharing an edge
        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        elements = torch.tensor([[0, 1, 2], [0, 2, 3]])
        # Boundary edges: (0,1), (1,2), (2,3), (3,0) - not (0,2) which is interior
        boundary_facets = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_facets=boundary_facets,
            facet_to_element=None,
            batch_size=[],
        )

        assert mesh.boundary_facets is not None
        assert mesh.boundary_facets.shape == (4, 2)

    def test_mesh_with_facet_to_element(self):
        """Create mesh with facet to element mapping."""
        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        elements = torch.tensor([[0, 1, 2], [0, 2, 3]])
        boundary_facets = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
        # Maps each boundary facet to its element
        facet_to_element = torch.tensor([0, 0, 1, 1])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_facets=boundary_facets,
            facet_to_element=facet_to_element,
            batch_size=[],
        )

        assert mesh.facet_to_element is not None
        assert mesh.facet_to_element.shape == (4,)


class TestMeshTensorclass:
    """Tests for Mesh tensorclass functionality."""

    def test_device_movement(self):
        """Mesh can be moved between devices."""
        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        elements = torch.tensor([[0, 1, 2]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        # Move to CPU explicitly (should work even if already on CPU)
        mesh_cpu = mesh.cpu()
        assert mesh_cpu.vertices.device.type == "cpu"

    def test_dtype_conversion(self):
        """Mesh tensors can be converted to different dtypes."""
        from torchscience.geometry.mesh import Mesh

        vertices = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32
        )
        elements = torch.tensor([[0, 1, 2]])

        mesh = Mesh(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_facets=None,
            facet_to_element=None,
            batch_size=[],
        )

        mesh_double = mesh.to(torch.float64)
        assert mesh_double.vertices.dtype == torch.float64

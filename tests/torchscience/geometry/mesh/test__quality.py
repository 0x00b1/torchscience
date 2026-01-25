# tests/torchscience/geometry/mesh/test__quality.py
"""Tests for mesh quality metrics."""

import math

import pytest
import torch

from torchscience.geometry.mesh import Mesh


class TestMeshQualityImport:
    """Tests for mesh_quality import."""

    def test_mesh_quality_importable(self):
        """mesh_quality is importable from geometry.mesh."""
        from torchscience.geometry.mesh import mesh_quality

        assert mesh_quality is not None

    def test_mesh_quality_importable_from_geometry(self):
        """mesh_quality is importable from geometry."""
        from torchscience.geometry import mesh_quality

        assert mesh_quality is not None


class TestMeshQualityAspectRatio:
    """Tests for aspect ratio quality metric."""

    def test_aspect_ratio_equilateral_triangle(self):
        """Aspect ratio of equilateral triangle should be 1.0."""
        from torchscience.geometry.mesh import mesh_quality

        # Equilateral triangle with vertices at equal distances
        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, math.sqrt(3) / 2],
            ],
            dtype=torch.float64,
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

        quality = mesh_quality(mesh, metric="aspect_ratio")

        assert quality.shape == (1,)
        assert torch.allclose(
            quality, torch.tensor([1.0], dtype=torch.float64), atol=1e-10
        )

    def test_aspect_ratio_stretched_triangle(self):
        """Aspect ratio of stretched triangle should be > 1.0."""
        from torchscience.geometry.mesh import mesh_quality

        # Stretched triangle: long base, small height
        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [5.0, 0.1],
            ],
            dtype=torch.float64,
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

        quality = mesh_quality(mesh, metric="aspect_ratio")

        assert quality.shape == (1,)
        # Stretched triangle has high aspect ratio
        assert quality[0] > 1.0
        # Edge lengths: base=10, two sides~=sqrt(25+0.01)~=5.001
        # Aspect ratio: 10 / 5.001 ~ 2.0
        expected_ratio = 10.0 / math.sqrt(5.0**2 + 0.1**2)
        assert torch.allclose(
            quality,
            torch.tensor([expected_ratio], dtype=torch.float64),
            atol=1e-6,
        )

    def test_aspect_ratio_right_triangle(self):
        """Aspect ratio of 3-4-5 right triangle."""
        from torchscience.geometry.mesh import mesh_quality

        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [0.0, 4.0],
            ],
            dtype=torch.float64,
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

        quality = mesh_quality(mesh, metric="aspect_ratio")

        assert quality.shape == (1,)
        # Edge lengths: 3, 4, 5
        # Aspect ratio: 5 / 3 = 5/3
        expected_ratio = 5.0 / 3.0
        assert torch.allclose(
            quality,
            torch.tensor([expected_ratio], dtype=torch.float64),
            atol=1e-10,
        )

    def test_aspect_ratio_default_metric(self):
        """Default metric should be aspect_ratio."""
        from torchscience.geometry.mesh import mesh_quality

        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, math.sqrt(3) / 2],
            ],
            dtype=torch.float64,
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

        # No metric specified - should use aspect_ratio
        quality = mesh_quality(mesh)

        assert torch.allclose(
            quality, torch.tensor([1.0], dtype=torch.float64), atol=1e-10
        )


class TestMeshQualityMinAngle:
    """Tests for minimum angle quality metric."""

    def test_min_angle_equilateral_triangle(self):
        """Minimum angle of equilateral triangle should be 60 degrees."""
        from torchscience.geometry.mesh import mesh_quality

        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, math.sqrt(3) / 2],
            ],
            dtype=torch.float64,
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

        quality = mesh_quality(mesh, metric="min_angle")

        assert quality.shape == (1,)
        assert torch.allclose(
            quality, torch.tensor([60.0], dtype=torch.float64), atol=1e-10
        )

    def test_min_angle_right_triangle(self):
        """Minimum angle of 45-45-90 right triangle should be 45 degrees."""
        from torchscience.geometry.mesh import mesh_quality

        # Isoceles right triangle
        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float64,
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

        quality = mesh_quality(mesh, metric="min_angle")

        assert quality.shape == (1,)
        assert torch.allclose(
            quality, torch.tensor([45.0], dtype=torch.float64), atol=1e-10
        )

    def test_min_angle_345_right_triangle(self):
        """Minimum angle of 3-4-5 right triangle."""
        from torchscience.geometry.mesh import mesh_quality

        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [0.0, 4.0],
            ],
            dtype=torch.float64,
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

        quality = mesh_quality(mesh, metric="min_angle")

        assert quality.shape == (1,)
        # Angles: arctan(3/4) ~ 36.87 degrees (smallest)
        expected_angle = math.degrees(math.atan(3.0 / 4.0))
        assert torch.allclose(
            quality,
            torch.tensor([expected_angle], dtype=torch.float64),
            atol=1e-10,
        )


class TestMeshQualityQuads:
    """Tests for quality metrics with quad elements."""

    def test_aspect_ratio_square_quad(self):
        """Aspect ratio of square quad should be 1.0."""
        from torchscience.geometry.mesh import mesh_quality

        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float64,
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

        quality = mesh_quality(mesh, metric="aspect_ratio")

        assert quality.shape == (1,)
        assert torch.allclose(
            quality, torch.tensor([1.0], dtype=torch.float64), atol=1e-10
        )

    def test_aspect_ratio_stretched_quad(self):
        """Aspect ratio of stretched quad should be > 1.0."""
        from torchscience.geometry.mesh import mesh_quality

        # Rectangle: width=4, height=1
        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float64,
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

        quality = mesh_quality(mesh, metric="aspect_ratio")

        assert quality.shape == (1,)
        # Edges: 4, 1, 4, 1 -> aspect ratio = 4/1 = 4.0
        assert torch.allclose(
            quality, torch.tensor([4.0], dtype=torch.float64), atol=1e-10
        )

    def test_min_angle_square_quad(self):
        """Minimum angle of square quad should be 90 degrees."""
        from torchscience.geometry.mesh import mesh_quality

        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float64,
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

        quality = mesh_quality(mesh, metric="min_angle")

        assert quality.shape == (1,)
        assert torch.allclose(
            quality, torch.tensor([90.0], dtype=torch.float64), atol=1e-10
        )

    def test_min_angle_parallelogram_quad(self):
        """Minimum angle of parallelogram quad."""
        from torchscience.geometry.mesh import mesh_quality

        # Parallelogram with 60 and 120 degree angles
        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.5, math.sqrt(3) / 2],
                [0.5, math.sqrt(3) / 2],
            ],
            dtype=torch.float64,
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

        quality = mesh_quality(mesh, metric="min_angle")

        assert quality.shape == (1,)
        assert torch.allclose(
            quality, torch.tensor([60.0], dtype=torch.float64), atol=1e-10
        )


class TestMeshQualityOutputShape:
    """Tests for output shape of quality metrics."""

    def test_output_shape_multiple_triangles(self):
        """Output shape should match num_elements for triangles."""
        from torchscience.geometry.mesh import mesh_quality, rectangle_mesh

        mesh = rectangle_mesh(
            2, 2, [[0.0, 1.0], [0.0, 1.0]], element_type="triangle"
        )
        # 2x2 grid of triangles = 8 triangles

        quality = mesh_quality(mesh, metric="aspect_ratio")

        assert quality.shape == (mesh.num_elements,)
        assert quality.shape == (8,)

    def test_output_shape_multiple_quads(self):
        """Output shape should match num_elements for quads."""
        from torchscience.geometry.mesh import mesh_quality, rectangle_mesh

        mesh = rectangle_mesh(
            3, 2, [[0.0, 1.0], [0.0, 1.0]], element_type="quad"
        )
        # 3x2 grid of quads = 6 quads

        quality = mesh_quality(mesh, metric="aspect_ratio")

        assert quality.shape == (mesh.num_elements,)
        assert quality.shape == (6,)


class TestMeshQualityInvalidInput:
    """Tests for error handling with invalid input."""

    def test_invalid_metric_raises_error(self):
        """Invalid metric should raise ValueError."""
        from torchscience.geometry.mesh import mesh_quality

        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, math.sqrt(3) / 2],
            ],
            dtype=torch.float64,
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

        with pytest.raises(ValueError, match="metric"):
            mesh_quality(mesh, metric="invalid_metric")


class TestMeshQualityDtypeDevice:
    """Tests for dtype and device handling."""

    def test_preserves_dtype_float32(self):
        """Output dtype should match input vertices dtype."""
        from torchscience.geometry.mesh import mesh_quality

        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, math.sqrt(3) / 2],
            ],
            dtype=torch.float32,
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

        quality = mesh_quality(mesh, metric="aspect_ratio")

        assert quality.dtype == torch.float32

    def test_preserves_device(self):
        """Output device should match input vertices device."""
        from torchscience.geometry.mesh import mesh_quality

        vertices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, math.sqrt(3) / 2],
            ],
            dtype=torch.float64,
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

        quality = mesh_quality(mesh, metric="aspect_ratio")

        assert quality.device == vertices.device

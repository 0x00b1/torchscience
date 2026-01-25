# tests/torchscience/geometry/mesh/test__delaunay.py
"""Tests for delaunay_triangulation function."""

import torch


class TestDelaunayTriangulationImport:
    """Tests for delaunay_triangulation import."""

    def test_delaunay_triangulation_importable(self):
        """delaunay_triangulation is importable from geometry.mesh."""
        from torchscience.geometry.mesh import delaunay_triangulation

        assert delaunay_triangulation is not None

    def test_delaunay_triangulation_importable_from_geometry(self):
        """delaunay_triangulation is importable from geometry."""
        from torchscience.geometry import delaunay_triangulation

        assert delaunay_triangulation is not None


class TestDelaunayTriangulationBasic:
    """Tests for basic Delaunay triangulation functionality."""

    def test_square_4_points(self):
        """Create triangulation from 4 points forming a square."""
        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )

        mesh = delaunay_triangulation(points)

        # Should produce a valid triangle mesh
        assert mesh.element_type == "triangle"
        # 4 points, 2 triangles (square split diagonally)
        assert mesh.num_elements == 2
        assert mesh.num_vertices == 4
        assert mesh.nodes_per_element == 3

    def test_pentagon_5_points(self):
        """Create triangulation from 5 points forming a pentagon."""
        import math

        from torchscience.geometry.mesh import delaunay_triangulation

        # Regular pentagon vertices
        angles = torch.linspace(0, 2 * math.pi, 6)[:-1]  # 5 points
        points = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

        mesh = delaunay_triangulation(points)

        # Convex hull of n points has n-2 triangles in triangulation
        # Pentagon has 5-2 = 3 triangles
        assert mesh.element_type == "triangle"
        assert mesh.num_elements == 3
        assert mesh.num_vertices == 5

    def test_triangle_3_points(self):
        """Create triangulation from 3 points forming a single triangle."""
        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ]
        )

        mesh = delaunay_triangulation(points)

        assert mesh.element_type == "triangle"
        assert mesh.num_elements == 1
        assert mesh.num_vertices == 3


class TestDelaunayTriangulationValidMesh:
    """Tests for mesh validity."""

    def test_all_input_points_are_vertices(self):
        """All input points appear as vertices in the output mesh."""
        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )

        mesh = delaunay_triangulation(points)

        # All input points should be present in vertices
        for point in points:
            distances = torch.norm(mesh.vertices - point, dim=1)
            assert torch.any(distances < 1e-6), (
                f"Point {point} not found in mesh"
            )

    def test_valid_element_connectivity(self):
        """Element indices are valid vertex indices."""
        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.rand(10, 2)

        mesh = delaunay_triangulation(points)

        # All element indices should be valid
        assert mesh.elements.min() >= 0
        assert mesh.elements.max() < mesh.num_vertices

        # Each element should have 3 unique vertices
        for i in range(mesh.num_elements):
            element = mesh.elements[i]
            assert len(torch.unique(element)) == 3

    def test_no_duplicate_vertices(self):
        """Output mesh has no duplicate vertices."""
        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.rand(15, 2)

        mesh = delaunay_triangulation(points)

        # Check for duplicates
        for i in range(mesh.num_vertices):
            for j in range(i + 1, mesh.num_vertices):
                dist = torch.norm(mesh.vertices[i] - mesh.vertices[j])
                assert dist > 1e-10, "Found duplicate vertices"


class TestDelaunayProperty:
    """Tests for the Delaunay property (circumcircle test)."""

    def test_delaunay_criterion_random_points(self):
        """No point lies inside any triangle's circumcircle."""
        from torchscience.geometry.mesh import delaunay_triangulation

        torch.manual_seed(42)
        points = torch.rand(20, 2, dtype=torch.float64)

        mesh = delaunay_triangulation(points)

        # For each triangle, check that no other point is inside its circumcircle
        for i in range(mesh.num_elements):
            tri_indices = mesh.elements[i]
            v0 = mesh.vertices[tri_indices[0]]
            v1 = mesh.vertices[tri_indices[1]]
            v2 = mesh.vertices[tri_indices[2]]

            # Compute circumcircle center and radius
            center, radius = _compute_circumcircle(v0, v1, v2)

            # Check no other vertex is strictly inside the circumcircle
            for j in range(mesh.num_vertices):
                if j in tri_indices.tolist():
                    continue  # Skip vertices of this triangle

                vertex = mesh.vertices[j]
                dist = torch.norm(vertex - center)

                # Allow small tolerance for numerical stability
                assert dist >= radius - 1e-9, (
                    f"Vertex {j} is inside circumcircle of triangle {i}"
                )

    def test_delaunay_criterion_structured_points(self):
        """Delaunay criterion holds for structured grid points."""
        from torchscience.geometry.mesh import delaunay_triangulation

        # Create a simple 3x3 grid
        x = torch.linspace(0, 1, 3, dtype=torch.float64)
        y = torch.linspace(0, 1, 3, dtype=torch.float64)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        mesh = delaunay_triangulation(points)

        # Check Delaunay property
        for i in range(mesh.num_elements):
            tri_indices = mesh.elements[i]
            v0 = mesh.vertices[tri_indices[0]]
            v1 = mesh.vertices[tri_indices[1]]
            v2 = mesh.vertices[tri_indices[2]]

            center, radius = _compute_circumcircle(v0, v1, v2)

            for j in range(mesh.num_vertices):
                if j in tri_indices.tolist():
                    continue

                vertex = mesh.vertices[j]
                dist = torch.norm(vertex - center)
                assert dist >= radius - 1e-9


class TestDelaunayTriangleCount:
    """Tests for triangle count in Delaunay triangulation."""

    def test_convex_triangle_count(self):
        """Convex point set has approximately 2n - 5 triangles."""
        from torchscience.geometry.mesh import delaunay_triangulation

        torch.manual_seed(123)
        # Random points in unit square (mostly convex-ish)
        n = 25
        points = torch.rand(n, 2)

        mesh = delaunay_triangulation(points)

        # For n points in general position, Delaunay has at most 2n - 5 triangles
        # (for points on convex hull) and at least n - 2 (if all collinear won't work)
        # The exact count depends on convex hull size h:
        # triangles = 2n - h - 2
        assert mesh.num_elements >= 1
        assert mesh.num_elements <= 2 * n - 5 + 1  # +1 for tolerance


class TestDelaunayCollinearPoints:
    """Tests for handling collinear points."""

    def test_collinear_points_3(self):
        """Collinear 3 points produce no triangles (degenerate case)."""
        import pytest

        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        )

        # Collinear points cannot form a valid triangulation
        # Should either raise an error or return empty mesh
        with pytest.raises((ValueError, RuntimeError)):
            delaunay_triangulation(points)

    def test_nearly_collinear_points(self):
        """Nearly collinear points are handled gracefully."""
        from torchscience.geometry.mesh import delaunay_triangulation

        # Points with a very small perpendicular offset
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1e-10],  # Nearly collinear
                [2.0, 0.0],
            ],
            dtype=torch.float64,
        )

        # This should either succeed with a degenerate triangle
        # or raise an appropriate error
        try:
            mesh = delaunay_triangulation(points)
            # If it succeeds, check that we get a valid mesh
            assert mesh.element_type == "triangle"
        except (ValueError, RuntimeError):
            pass  # Also acceptable


class TestDelaunayDtypeDevice:
    """Tests for dtype and device handling."""

    def test_float64_dtype(self):
        """Float64 points produce float64 vertices."""
        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.rand(10, 2, dtype=torch.float64)

        mesh = delaunay_triangulation(points)

        assert mesh.vertices.dtype == torch.float64

    def test_float32_dtype(self):
        """Float32 points produce float32 vertices."""
        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.rand(10, 2, dtype=torch.float32)

        mesh = delaunay_triangulation(points)

        assert mesh.vertices.dtype == torch.float32

    def test_elements_int64_dtype(self):
        """Elements tensor has int64 dtype."""
        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.rand(10, 2)

        mesh = delaunay_triangulation(points)

        assert mesh.elements.dtype == torch.int64

    def test_cpu_device(self):
        """CPU points produce CPU mesh."""
        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.rand(10, 2, device="cpu")

        mesh = delaunay_triangulation(points)

        assert mesh.vertices.device.type == "cpu"
        assert mesh.elements.device.type == "cpu"


class TestDelaunayInputValidation:
    """Tests for input validation."""

    def test_less_than_3_points_raises(self):
        """Less than 3 points raises an error."""
        import pytest

        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(ValueError, match="at least 3 points"):
            delaunay_triangulation(points)

    def test_1d_points_raises(self):
        """1D points tensor raises an error."""
        import pytest

        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.rand(10)

        with pytest.raises(ValueError, match="2D"):
            delaunay_triangulation(points)

    def test_3d_points_raises(self):
        """3D points raises an error (only 2D supported)."""
        import pytest

        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.rand(10, 3)

        with pytest.raises(ValueError, match="2D"):
            delaunay_triangulation(points)


class TestDelaunayDuplicatePoints:
    """Tests for handling duplicate input points."""

    def test_duplicate_points_handled(self):
        """Duplicate points in input are handled gracefully."""
        from torchscience.geometry.mesh import delaunay_triangulation

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],  # Duplicate of first point
            ]
        )

        # Should either:
        # 1. Remove duplicate and triangulate 4 unique points
        # 2. Raise an informative error
        try:
            mesh = delaunay_triangulation(points)
            # If successful, should have triangulated unique points
            assert mesh.element_type == "triangle"
            assert mesh.num_vertices <= 4  # At most 4 unique vertices
        except (ValueError, RuntimeError):
            pass  # Also acceptable


# Helper function for circumcircle computation
def _compute_circumcircle(v0, v1, v2):
    """Compute circumcircle center and radius for a triangle.

    Parameters
    ----------
    v0, v1, v2 : Tensor
        Triangle vertices, shape (2,).

    Returns
    -------
    center : Tensor
        Circumcircle center, shape (2,).
    radius : float
        Circumcircle radius.
    """
    # Use the formula for circumcircle of a triangle
    ax, ay = v0[0], v0[1]
    bx, by = v1[0], v1[1]
    cx, cy = v2[0], v2[1]

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    if abs(d) < 1e-12:
        # Degenerate triangle (collinear points)
        center = (v0 + v1 + v2) / 3
        radius = torch.tensor(float("inf"))
        return center, radius

    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / d
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / d

    center = torch.stack([ux, uy])
    radius = torch.norm(v0 - center)

    return center, radius

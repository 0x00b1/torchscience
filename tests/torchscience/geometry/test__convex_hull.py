# tests/torchscience/geometry/test__convex_hull.py
import pytest
import torch

import torchscience  # noqa: F401 - needed to register C++ operators


class TestConvexHullImport:
    """Tests for convex_hull import."""

    def test_convex_hull_importable(self):
        """convex_hull function is importable."""
        from torchscience.geometry import convex_hull

        assert callable(convex_hull)

    def test_convex_hull_class_importable(self):
        """ConvexHull class is importable."""
        from torchscience.geometry import ConvexHull

        assert ConvexHull is not None


class TestConvexHullSchema:
    """Tests for convex_hull operator schema."""

    def test_operator_exists(self):
        """convex_hull operator is registered."""
        assert hasattr(torch.ops.torchscience, "convex_hull")


class TestConvexHullMeta:
    """Tests for convex_hull meta tensor support."""

    def test_meta_shape_inference(self):
        """Meta tensors produce correct output shapes."""
        points = torch.randn(100, 3, device="meta")
        result = torch.ops.torchscience.convex_hull(points)

        # Unpack tuple
        (
            vertices,
            simplices,
            neighbors,
            equations,
            area,
            volume,
            n_vertices,
            n_facets,
        ) = result

        # Check shapes (max_vertices and max_facets are upper bounds)
        assert vertices.dim() == 1  # (max_vertices,)
        assert simplices.dim() == 2  # (max_facets, n_dims)
        assert simplices.shape[1] == 3  # n_dims
        assert equations.dim() == 2  # (max_facets, n_dims + 1)
        assert equations.shape[1] == 4  # n_dims + 1
        assert area.shape == ()  # scalar
        assert volume.shape == ()  # scalar

    def test_meta_batched_shape_inference(self):
        """Batched meta tensors produce correct output shapes."""
        points = torch.randn(4, 100, 3, device="meta")
        result = torch.ops.torchscience.convex_hull(points)

        (
            vertices,
            simplices,
            neighbors,
            equations,
            area,
            volume,
            n_vertices,
            n_facets,
        ) = result

        assert vertices.shape[0] == 4  # batch
        assert area.shape == (4,)
        assert volume.shape == (4,)


class TestConvexHull2D:
    """Tests for 2D convex hull."""

    def test_square_hull(self):
        """Unit square hull has 4 vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.5],  # interior point
            ]
        )
        result = torch.ops.torchscience.convex_hull(points)
        (
            vertices,
            simplices,
            neighbors,
            equations,
            area,
            volume,
            n_vertices,
            n_facets,
        ) = result

        assert n_vertices.item() == 4
        assert n_facets.item() == 4  # 4 edges in 2D

    def test_triangle_hull(self):
        """Triangle hull has 3 vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ]
        )
        result = torch.ops.torchscience.convex_hull(points)
        _, _, _, _, _, _, n_vertices, n_facets = result

        assert n_vertices.item() == 3
        assert n_facets.item() == 3

    def test_area_square(self):
        """Unit square has area 1.0."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        result = torch.ops.torchscience.convex_hull(points)
        _, _, _, _, _, volume, _, _ = result

        # In 2D, "volume" is area
        torch.testing.assert_close(volume, torch.tensor(1.0))


class TestConvexHullPythonAPI:
    """Tests for convex_hull Python wrapper returning ConvexHull tensorclass."""

    def test_returns_convex_hull_class(self):
        """convex_hull returns ConvexHull tensorclass."""
        from torchscience.geometry import ConvexHull, convex_hull

        points = torch.rand(10, 2)
        hull = convex_hull(points)
        assert isinstance(hull, ConvexHull)

    def test_contains_input_points(self):
        """ConvexHull stores original input points."""
        from torchscience.geometry import convex_hull

        points = torch.rand(10, 2)
        hull = convex_hull(points)
        torch.testing.assert_close(hull.points, points)

    def test_area_property(self):
        """ConvexHull.area returns perimeter in 2D."""
        from torchscience.geometry import convex_hull

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        hull = convex_hull(points)
        # Perimeter of unit square is 4.0
        torch.testing.assert_close(hull.area, torch.tensor(4.0))

    def test_volume_property(self):
        """ConvexHull.volume returns area in 2D."""
        from torchscience.geometry import convex_hull

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        hull = convex_hull(points)
        # Area of unit square is 1.0
        torch.testing.assert_close(hull.volume, torch.tensor(1.0))

    def test_n_dims_property(self):
        """ConvexHull.n_dims returns dimensionality."""
        from torchscience.geometry import convex_hull

        points_2d = torch.rand(10, 2)
        hull_2d = convex_hull(points_2d)
        assert hull_2d.n_dims == 2

        points_3d = torch.rand(10, 3)
        hull_3d = convex_hull(points_3d)
        assert hull_3d.n_dims == 3

    def test_vertices_are_indices(self):
        """ConvexHull.vertices are point indices on the hull."""
        from torchscience.geometry import convex_hull

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.5],  # interior
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        hull = convex_hull(points)
        # Should have 4 vertices (indices 0, 1, 3, 4)
        assert hull.n_vertices.item() == 4
        # Check indices are in valid range
        assert hull.vertices[:4].min() >= 0
        assert hull.vertices[:4].max() <= 4


class TestConvexHull3D:
    """Tests for 3D convex hull."""

    def test_tetrahedron_hull(self):
        """Tetrahedron hull has 4 vertices and 4 faces."""
        from torchscience.geometry import convex_hull

        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ]
        )
        hull = convex_hull(points)
        assert hull.n_vertices.item() == 4
        assert hull.n_facets.item() == 4

    def test_cube_hull(self):
        """Cube hull has 8 vertices and 12 faces (triangulated)."""
        from torchscience.geometry import convex_hull

        points = torch.tensor(
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
        hull = convex_hull(points)
        assert hull.n_vertices.item() == 8
        # A triangulated cube has 12 triangular faces (2 per square face)
        assert hull.n_facets.item() == 12

    def test_volume_tetrahedron(self):
        """Unit tetrahedron has known volume."""
        from torchscience.geometry import convex_hull

        # Regular tetrahedron with vertices at corners of unit cube
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        hull = convex_hull(points)
        # Volume of this tetrahedron is 1/3
        torch.testing.assert_close(
            hull.volume, torch.tensor(1.0 / 3.0), atol=1e-5, rtol=1e-5
        )


class TestScipyComparison:
    """Tests comparing torchscience convex_hull with scipy.spatial.ConvexHull."""

    def test_2d_random_vertices_match_scipy(self):
        """2D random hull vertices match scipy."""
        scipy_spatial = pytest.importorskip("scipy.spatial")

        from torchscience.geometry import convex_hull

        torch.manual_seed(42)
        points = torch.rand(50, 2)

        # torchscience
        hull = convex_hull(points)
        ts_vertices = set(hull.vertices[: hull.n_vertices.item()].tolist())

        # scipy
        scipy_hull = scipy_spatial.ConvexHull(points.numpy())
        scipy_vertices = set(scipy_hull.vertices.tolist())

        assert ts_vertices == scipy_vertices

    def test_2d_random_area_matches_scipy(self):
        """2D random hull area matches scipy."""
        scipy_spatial = pytest.importorskip("scipy.spatial")

        from torchscience.geometry import convex_hull

        torch.manual_seed(123)
        points = torch.rand(100, 2)

        # torchscience (volume = area in 2D)
        hull = convex_hull(points)

        # scipy
        scipy_hull = scipy_spatial.ConvexHull(points.numpy())

        torch.testing.assert_close(
            hull.volume,
            torch.tensor(scipy_hull.volume, dtype=hull.volume.dtype),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_3d_random_vertices_match_scipy(self):
        """3D random hull vertices match scipy."""
        scipy_spatial = pytest.importorskip("scipy.spatial")

        from torchscience.geometry import convex_hull

        torch.manual_seed(42)
        points = torch.rand(50, 3)

        # torchscience
        hull = convex_hull(points)
        ts_vertices = set(hull.vertices[: hull.n_vertices.item()].tolist())

        # scipy
        scipy_hull = scipy_spatial.ConvexHull(points.numpy())
        scipy_vertices = set(scipy_hull.vertices.tolist())

        assert ts_vertices == scipy_vertices

    def test_3d_random_volume_matches_scipy(self):
        """3D random hull volume matches scipy."""
        scipy_spatial = pytest.importorskip("scipy.spatial")

        from torchscience.geometry import convex_hull

        torch.manual_seed(123)
        points = torch.rand(100, 3)

        # torchscience
        hull = convex_hull(points)

        # scipy
        scipy_hull = scipy_spatial.ConvexHull(points.numpy())

        torch.testing.assert_close(
            hull.volume,
            torch.tensor(scipy_hull.volume, dtype=hull.volume.dtype),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_3d_random_surface_area_matches_scipy(self):
        """3D random hull surface area matches scipy."""
        scipy_spatial = pytest.importorskip("scipy.spatial")

        from torchscience.geometry import convex_hull

        torch.manual_seed(456)
        points = torch.rand(80, 3)

        # torchscience (area = surface area in 3D)
        hull = convex_hull(points)

        # scipy
        scipy_hull = scipy_spatial.ConvexHull(points.numpy())

        torch.testing.assert_close(
            hull.area,
            torch.tensor(scipy_hull.area, dtype=hull.area.dtype),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_3d_facet_count_matches_scipy(self):
        """3D hull facet count matches scipy."""
        scipy_spatial = pytest.importorskip("scipy.spatial")

        from torchscience.geometry import convex_hull

        torch.manual_seed(789)
        points = torch.rand(60, 3)

        # torchscience
        hull = convex_hull(points)

        # scipy
        scipy_hull = scipy_spatial.ConvexHull(points.numpy())

        assert hull.n_facets.item() == len(scipy_hull.simplices)

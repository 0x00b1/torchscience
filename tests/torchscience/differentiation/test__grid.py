"""Tests for RegularGrid and IrregularMesh dataclasses."""

import torch

from torchscience.differentiation import (
    IrregularMesh,
    RegularGrid,
    biharmonic,
    curl,
    derivative,
    divergence,
    gradient,
    hessian,
    jacobian,
    laplacian,
)


class TestRegularGrid:
    """Tests for RegularGrid tensorclass."""

    def test_create_1d_grid(self):
        """Create a 1D regular grid."""
        grid = RegularGrid(
            origin=torch.tensor([0.0]),
            spacing=torch.tensor([0.1]),
            shape=(11,),
            boundary="periodic",
        )
        assert grid.ndim == 1
        assert grid.n_points == 11

    def test_create_2d_grid(self):
        """Create a 2D regular grid."""
        grid = RegularGrid(
            origin=torch.tensor([0.0, 0.0]),
            spacing=torch.tensor([0.1, 0.2]),
            shape=(11, 21),
            boundary="replicate",
        )
        assert grid.ndim == 2
        assert grid.n_points == 11 * 21

    def test_grid_points_property(self):
        """Grid points property returns coordinates."""
        grid = RegularGrid(
            origin=torch.tensor([0.0]),
            spacing=torch.tensor([0.5]),
            shape=(5,),
            boundary="periodic",
        )
        points = grid.points
        expected = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]])
        torch.testing.assert_close(points, expected)

    def test_grid_2d_points(self):
        """2D grid points are correct."""
        grid = RegularGrid(
            origin=torch.tensor([0.0, 0.0]),
            spacing=torch.tensor([1.0, 1.0]),
            shape=(2, 3),
            boundary="replicate",
        )
        points = grid.points
        assert points.shape == (6, 2)
        # Points should be (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)


class TestIrregularMesh:
    """Tests for IrregularMesh dataclass."""

    def test_create_from_points(self):
        """Create mesh from point coordinates."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.866],  # equilateral triangle
            ]
        )
        mesh = IrregularMesh(points=points)
        assert mesh.ndim == 2
        assert mesh.n_points == 3

    def test_neighbors(self):
        """Find k-nearest neighbors."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )
        mesh = IrregularMesh(points=points)
        neighbors = mesh.neighbors(k=2)
        # Each point has 2 nearest neighbors
        assert neighbors.shape == (4, 2)
        # Point 0's nearest neighbors should be 1 and 2
        assert 1 in neighbors[0].tolist()

    def test_from_regular_grid(self):
        """Convert RegularGrid to IrregularMesh."""
        grid = RegularGrid(
            origin=torch.tensor([0.0, 0.0]),
            spacing=torch.tensor([1.0, 1.0]),
            shape=(3, 3),
            boundary="replicate",
        )
        mesh = grid.to_mesh()
        assert isinstance(mesh, IrregularMesh)
        assert mesh.n_points == 9


class TestGradientWithGrid:
    """Tests for gradient with grid parameter."""

    def test_gradient_with_regular_grid(self):
        """Gradient uses grid spacing."""
        grid = RegularGrid(
            origin=torch.tensor([0.0, 0.0]),
            spacing=torch.tensor([0.1, 0.1]),
            shape=(21, 21),
            boundary="replicate",
        )

        # Create linear field: f(x, y) = 3x + 2y
        x = torch.linspace(0, 2, 21)
        y = torch.linspace(0, 2, 21)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = 3 * X + 2 * Y

        grad = gradient(field, grid=grid)

        # df/dx = 3, df/dy = 2
        assert grad.shape == (2, 21, 21)
        torch.testing.assert_close(
            grad[0, 2:-2, 2:-2],
            torch.full((17, 17), 3.0),
            rtol=0.05,
            atol=0.01,
        )
        torch.testing.assert_close(
            grad[1, 2:-2, 2:-2],
            torch.full((17, 17), 2.0),
            rtol=0.05,
            atol=0.01,
        )

    def test_gradient_grid_overrides_dx(self):
        """Grid parameter takes precedence over dx."""
        grid = RegularGrid(
            origin=torch.tensor([0.0]),
            spacing=torch.tensor([0.5]),  # dx=0.5
            shape=(11,),
            boundary="replicate",
        )

        field = torch.linspace(0, 5, 11)  # f = x, so df/dx = 1

        # dx=1.0 should be ignored when grid is provided
        grad = gradient(field, dx=1.0, grid=grid)

        # With dx=0.5, gradient should be 1.0
        torch.testing.assert_close(
            grad[0, 2:-2],
            torch.full((7,), 1.0),
            rtol=0.05,
            atol=0.01,
        )


class TestAllOperatorsWithGrid:
    """Test grid parameter for all operators."""

    def test_derivative_with_grid(self):
        """Derivative uses grid spacing."""
        grid = RegularGrid(
            origin=torch.tensor([0.0, 0.0]),
            spacing=torch.tensor([0.1, 0.1]),
            shape=(21, 21),
            boundary="replicate",
        )
        field = torch.randn(21, 21)
        result = derivative(field, dim=-1, grid=grid)
        assert result.shape == (21, 21)

    def test_laplacian_with_grid(self):
        """Laplacian uses grid spacing."""
        grid = RegularGrid(
            origin=torch.tensor([0.0, 0.0]),
            spacing=torch.tensor([0.1, 0.1]),
            shape=(21, 21),
            boundary="replicate",
        )
        field = torch.randn(21, 21)
        result = laplacian(field, grid=grid)
        assert result.shape == (21, 21)

    def test_hessian_with_grid(self):
        """Hessian uses grid spacing."""
        grid = RegularGrid(
            origin=torch.tensor([0.0, 0.0]),
            spacing=torch.tensor([0.1, 0.1]),
            shape=(21, 21),
            boundary="replicate",
        )
        field = torch.randn(21, 21)
        result = hessian(field, grid=grid)
        assert result.shape == (2, 2, 21, 21)

    def test_biharmonic_with_grid(self):
        """Biharmonic uses grid spacing."""
        grid = RegularGrid(
            origin=torch.tensor([0.0, 0.0]),
            spacing=torch.tensor([0.1, 0.1]),
            shape=(21, 21),
            boundary="replicate",
        )
        field = torch.randn(21, 21)
        result = biharmonic(field, grid=grid)
        assert result.shape == (21, 21)

    def test_divergence_with_grid(self):
        """Divergence uses grid spacing."""
        grid = RegularGrid(
            origin=torch.tensor([0.0, 0.0]),
            spacing=torch.tensor([0.1, 0.1]),
            shape=(21, 21),
            boundary="replicate",
        )
        field = torch.randn(2, 21, 21)
        result = divergence(field, grid=grid)
        assert result.shape == (21, 21)

    def test_curl_with_grid(self):
        """Curl uses grid spacing."""
        grid = RegularGrid(
            origin=torch.tensor([0.0, 0.0, 0.0]),
            spacing=torch.tensor([0.1, 0.1, 0.1]),
            shape=(8, 8, 8),
            boundary="replicate",
        )
        field = torch.randn(3, 8, 8, 8)
        result = curl(field, grid=grid)
        assert result.shape == (3, 8, 8, 8)

    def test_jacobian_with_grid(self):
        """Jacobian uses grid spacing."""
        grid = RegularGrid(
            origin=torch.tensor([0.0, 0.0]),
            spacing=torch.tensor([0.1, 0.1]),
            shape=(21, 21),
            boundary="replicate",
        )
        field = torch.randn(2, 21, 21)
        result = jacobian(field, grid=grid)
        assert result.shape == (2, 2, 21, 21)

"""Tests for RegularGrid dataclass."""

import torch

from torchscience.differentiation import RegularGrid


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

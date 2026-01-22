"""Grid and mesh abstractions for differentiation operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class RegularGrid:
    """Regular Cartesian grid with uniform spacing.

    Parameters
    ----------
    origin : Tensor
        Grid origin coordinates, shape (ndim,).
    spacing : Tensor
        Grid spacing per dimension (dx, dy, ...), shape (ndim,).
    shape : Tuple[int, ...]
        Number of grid points per dimension.
    boundary : str
        Boundary condition: 'periodic', 'replicate', 'zeros', 'reflect'.

    Examples
    --------
    >>> grid = RegularGrid(
    ...     origin=torch.tensor([0.0, 0.0]),
    ...     spacing=torch.tensor([0.1, 0.1]),
    ...     shape=(10, 10),
    ...     boundary="periodic",
    ... )
    >>> grid.ndim
    2
    """

    origin: Tensor
    spacing: Tensor
    shape: Tuple[int, ...]
    boundary: str

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.shape)

    @property
    def n_points(self) -> int:
        """Total number of grid points."""
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def points(self) -> Tensor:
        """Compute grid point coordinates.

        Returns
        -------
        Tensor
            Grid coordinates, shape (n_points, ndim).
        """
        coords = []
        for i, n in enumerate(self.shape):
            coord = self.origin[i] + self.spacing[i] * torch.arange(
                n, dtype=self.origin.dtype, device=self.origin.device
            )
            coords.append(coord)

        grids = torch.meshgrid(*coords, indexing="ij")
        return torch.stack([g.flatten() for g in grids], dim=-1)

    def dx(self, dim: int) -> Tensor:
        """Get spacing for a specific dimension."""
        return self.spacing[dim]

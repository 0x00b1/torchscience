"""Finite difference stencil data structure."""

from __future__ import annotations

from typing import Tuple

import torch
from tensordict import tensorclass
from torch import Tensor


@tensorclass
class FiniteDifferenceStencil:
    """N-dimensional finite difference stencil.

    A stencil represents a weighted sum of function values at offset grid points.
    The sparse representation stores only non-zero coefficients.

    Attributes
    ----------
    offsets : Tensor
        Integer offsets from center point, shape (n_points, ndim).
        Each row is an offset vector [i, j, ...] meaning grid[..., x+i, y+j, ...].
    coeffs : Tensor
        Coefficients for each offset, shape (n_points,).
        The derivative is sum(coeffs[k] * f[center + offsets[k]]) / dx^order.
    derivative : Tuple[int, ...]
        Derivative order per dimension. (2, 0) means d^2/dx^2.
        (1, 1) means mixed partial d^2/dxdy.
    accuracy : int
        Accuracy order of the stencil (error is O(dx^accuracy)).

    Examples
    --------
    Central second derivative in 1D (3-point stencil):

    >>> stencil = FiniteDifferenceStencil(
    ...     offsets=torch.tensor([[-1], [0], [1]]),
    ...     coeffs=torch.tensor([1.0, -2.0, 1.0]),
    ...     derivative=(2,),
    ...     accuracy=2,
    ... )

    5-point Laplacian in 2D:

    >>> stencil = FiniteDifferenceStencil(
    ...     offsets=torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]),
    ...     coeffs=torch.tensor([-4.0, 1.0, 1.0, 1.0, 1.0]),
    ...     derivative=(2, 2),  # Represents d^2/dx^2 + d^2/dy^2
    ...     accuracy=2,
    ... )
    """

    offsets: Tensor  # (n_points, ndim), int64
    coeffs: Tensor  # (n_points,), float
    derivative: Tuple[int, ...]
    accuracy: int

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self.offsets.shape[-1]

    @property
    def n_points(self) -> int:
        """Number of stencil points."""
        return self.offsets.shape[0]

    @property
    def order(self) -> int:
        """Total derivative order (sum of orders per dimension)."""
        return sum(self.derivative)

    def to_dense(self, shape: Tuple[int, ...] | None = None) -> Tensor:
        """Convert sparse stencil to dense convolution kernel.

        Parameters
        ----------
        shape : tuple of int, optional
            Output kernel shape. If None, uses minimal bounding box.

        Returns
        -------
        Tensor
            Dense kernel suitable for torch.nn.functional.conv operations.
            Shape is (1, 1, *spatial_dims) for convolution.
        """
        # Compute minimal bounding box from offsets
        offsets = self.offsets
        min_offset = offsets.min(dim=0).values
        max_offset = offsets.max(dim=0).values

        if shape is None:
            kernel_shape = (max_offset - min_offset + 1).tolist()
        else:
            kernel_shape = list(shape)

        # Create dense kernel
        kernel = torch.zeros(
            kernel_shape, dtype=self.coeffs.dtype, device=self.coeffs.device
        )

        # Compute center offset (so that offset=0 maps to center of kernel)
        center = -min_offset

        for i in range(self.n_points):
            idx = tuple((offsets[i] + center).tolist())
            kernel[idx] = self.coeffs[i]

        # Add batch and channel dimensions for conv: (1, 1, *spatial)
        return kernel.unsqueeze(0).unsqueeze(0)

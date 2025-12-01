"""Convex hull implementation with tensorclass-based API."""

from __future__ import annotations

import torch
from tensordict import tensorclass
from torch import Tensor

import torchscience  # noqa: F401 - needed to register C++ operators


@tensorclass
class ConvexHull:
    """N-dimensional convex hull.

    As a tensorclass, ConvexHull supports:
    - Automatic batching: indexing with `hull[0]` or `hull[:2]`
    - Device movement: `hull.to("cuda")` or `hull.cuda()`
    - Serialization: `torch.save(hull, path)` / `torch.load(path)`

    Attributes
    ----------
    points : Tensor
        Original input points, shape (*batch, n_points, n_dims).
    vertices : Tensor
        Indices of vertices on hull boundary, shape (*batch, max_vertices).
        Padded with -1 for batch elements with fewer vertices.
    simplices : Tensor
        Facet vertex indices, shape (*batch, max_facets, n_dims).
        Each row contains n_dims vertex indices forming a facet.
    neighbors : Tensor
        Neighbor facet indices, shape (*batch, max_facets, n_dims).
    equations : Tensor
        Facet equations [normal, offset], shape (*batch, max_facets, n_dims + 1).
    _area : Tensor
        Cached surface area (perimeter in 2D), shape (*batch,).
    _volume : Tensor
        Cached volume (area in 2D), shape (*batch,).
    n_vertices : Tensor
        Actual vertex count per batch element, shape (*batch,).
    n_facets : Tensor
        Actual facet count per batch element, shape (*batch,).

    Notes
    -----
    Hull construction is NOT differentiable (discrete structure).
    Query methods (distance, project) ARE differentiable.
    """

    points: Tensor
    vertices: Tensor
    simplices: Tensor
    neighbors: Tensor
    equations: Tensor
    _area: Tensor
    _volume: Tensor
    n_vertices: Tensor
    n_facets: Tensor

    @property
    def area(self) -> Tensor:
        """Surface area (perimeter in 2D)."""
        return self._area

    @property
    def volume(self) -> Tensor:
        """Volume (area in 2D)."""
        return self._volume

    @property
    def n_dims(self) -> int:
        """Dimensionality of the hull."""
        return self.points.shape[-1]


def convex_hull(points: Tensor) -> ConvexHull:
    """Compute the convex hull of a point set.

    Parameters
    ----------
    points : Tensor
        Input points with shape (*batch, n_points, n_dims).
        Requires n_points >= n_dims + 1.

    Returns
    -------
    ConvexHull
        Tensorclass containing hull structure and methods.

    Raises
    ------
    ValueError
        If points has fewer than 2 dimensions.
    InsufficientPointsError
        If n_points < n_dims + 1.

    Notes
    -----
    Hull construction is NOT differentiable (discrete structure).
    Query methods (distance, project) ARE differentiable.

    Examples
    --------
    >>> points = torch.rand(100, 3)  # 100 points in 3D
    >>> hull = convex_hull(points)
    >>> hull.vertices  # indices of vertices on hull
    >>> hull.volume    # volume of hull

    >>> # Batched
    >>> points = torch.rand(8, 100, 3)  # batch of 8
    >>> hull = convex_hull(points)
    >>> hull.volume.shape  # (8,)
    """
    # Call C++ operator
    (
        vertices,
        simplices,
        neighbors,
        equations,
        area,
        volume,
        n_vertices,
        n_facets,
    ) = torch.ops.torchscience.convex_hull(points)

    return ConvexHull(
        points=points,
        vertices=vertices,
        simplices=simplices,
        neighbors=neighbors,
        equations=equations,
        _area=area,
        _volume=volume,
        n_vertices=n_vertices,
        n_facets=n_facets,
        batch_size=[],
    )

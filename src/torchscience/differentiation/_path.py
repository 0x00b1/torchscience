"""Path and Surface classes for line and surface integrals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import Tensor


@dataclass
class Path:
    """Discretized path for line integrals.

    Represents an ordered sequence of points in n-dimensional space.

    Parameters
    ----------
    points : Tensor
        Points along the path with shape (N, ndim).
    closed : bool
        Whether the path forms a closed loop. Default False.

    Examples
    --------
    >>> points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    >>> path = Path(points=points)
    >>> path.n_points
    3
    >>> path.total_length
    tensor(2.)
    """

    points: Tensor
    closed: bool = False

    @property
    def n_points(self) -> int:
        """Number of points in the path."""
        return self.points.shape[0]

    @property
    def ndim(self) -> int:
        """Spatial dimension of the path."""
        return self.points.shape[1]

    @property
    def tangents(self) -> Tensor:
        """Tangent vectors at each segment.

        Returns
        -------
        Tensor
            Shape (N-1, ndim) for open paths, (N, ndim) for closed paths.
        """
        if self.closed:
            next_points = torch.roll(self.points, -1, dims=0)
            return next_points - self.points
        else:
            return self.points[1:] - self.points[:-1]

    @property
    def segment_lengths(self) -> Tensor:
        """Length of each segment.

        Returns
        -------
        Tensor
            Shape (N-1,) for open paths, (N,) for closed paths.
        """
        return torch.linalg.norm(self.tangents, dim=-1)

    @property
    def total_length(self) -> Tensor:
        """Total arc length of the path.

        Returns
        -------
        Tensor
            Scalar tensor with the total arc length.
        """
        return self.segment_lengths.sum()

    @property
    def midpoints(self) -> Tensor:
        """Midpoints of each segment.

        Returns
        -------
        Tensor
            Shape (N-1, ndim) for open paths, (N, ndim) for closed paths.
        """
        if self.closed:
            next_points = torch.roll(self.points, -1, dims=0)
            return (self.points + next_points) / 2
        else:
            return (self.points[:-1] + self.points[1:]) / 2

    @staticmethod
    def from_parametric(
        func: Callable[[Tensor], Tensor],
        t: Tensor,
        closed: bool = False,
    ) -> "Path":
        """Create path from parametric function.

        Parameters
        ----------
        func : Callable
            Function f(t) -> (ndim,) returning point at parameter t.
        t : Tensor
            Parameter values (N,).
        closed : bool
            Whether path is closed.

        Returns
        -------
        Path
            Path with points sampled from the parametric function.

        Examples
        --------
        >>> import math
        >>> def circle(t):
        ...     return torch.stack([torch.cos(t), torch.sin(t)])
        >>> t = torch.linspace(0, 2 * math.pi, 100)
        >>> path = Path.from_parametric(circle, t, closed=True)
        """
        points = torch.stack([func(ti) for ti in t], dim=0)
        return Path(points=points, closed=closed)


@dataclass
class Surface:
    """Discretized surface for surface integrals.

    Represents a 2D surface embedded in n-dimensional space.

    Parameters
    ----------
    points : Tensor
        Surface points with shape (Nu, Nv, ndim).
    _normals : Tensor, optional
        Precomputed normal vectors.

    Examples
    --------
    >>> u = torch.linspace(0, 1, 10)
    >>> v = torch.linspace(0, 1, 10)
    >>> U, V = torch.meshgrid(u, v, indexing="ij")
    >>> points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
    >>> surface = Surface(points=points)
    >>> surface.shape
    (10, 10)
    """

    points: Tensor
    _normals: Tensor | None = None

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape (Nu, Nv)."""
        return (self.points.shape[0], self.points.shape[1])

    @property
    def ndim(self) -> int:
        """Spatial dimension."""
        return self.points.shape[-1]

    @property
    def normals(self) -> Tensor:
        """Surface normal vectors at each point.

        Computed from cross product of tangent vectors using central
        differences for interior points and forward/backward differences
        at boundaries.

        Returns
        -------
        Tensor
            Normal vectors with shape (Nu, Nv, ndim).

        Raises
        ------
        ValueError
            If the surface is not in 3D space.
        """
        if self._normals is not None:
            return self._normals

        if self.ndim != 3:
            raise ValueError(
                f"Normal computation requires 3D space, got {self.ndim}D"
            )

        # Tangent vectors in u and v directions using central differences
        du = torch.zeros_like(self.points)
        dv = torch.zeros_like(self.points)

        # u direction (central differences for interior, forward/backward at edges)
        du[1:-1] = (self.points[2:] - self.points[:-2]) / 2
        du[0] = self.points[1] - self.points[0]
        du[-1] = self.points[-1] - self.points[-2]

        # v direction (central differences for interior, forward/backward at edges)
        dv[:, 1:-1] = (self.points[:, 2:] - self.points[:, :-2]) / 2
        dv[:, 0] = self.points[:, 1] - self.points[:, 0]
        dv[:, -1] = self.points[:, -1] - self.points[:, -2]

        # Cross product for 3D
        normals = torch.cross(du, dv, dim=-1)
        norms = torch.linalg.norm(normals, dim=-1, keepdim=True)
        return normals / (norms + 1e-8)

    @property
    def area_elements(self) -> Tensor:
        """Area element |du x dv| at each point.

        Returns
        -------
        Tensor
            Area elements with shape (Nu, Nv).

        Raises
        ------
        ValueError
            If the surface is not in 3D space.
        """
        if self.ndim != 3:
            raise ValueError(
                f"Area computation requires 3D space, got {self.ndim}D"
            )

        du = torch.zeros_like(self.points)
        dv = torch.zeros_like(self.points)

        du[1:-1] = (self.points[2:] - self.points[:-2]) / 2
        du[0] = self.points[1] - self.points[0]
        du[-1] = self.points[-1] - self.points[-2]

        dv[:, 1:-1] = (self.points[:, 2:] - self.points[:, :-2]) / 2
        dv[:, 0] = self.points[:, 1] - self.points[:, 0]
        dv[:, -1] = self.points[:, -1] - self.points[:, -2]

        cross = torch.cross(du, dv, dim=-1)
        return torch.linalg.norm(cross, dim=-1)

    @staticmethod
    def from_parametric(
        func: Callable[[Tensor, Tensor], Tensor],
        u: Tensor,
        v: Tensor,
    ) -> "Surface":
        """Create surface from parametric function.

        Parameters
        ----------
        func : Callable
            Function f(u, v) -> (ndim,) returning point at parameters (u, v).
        u, v : Tensor
            Parameter grids (1D tensors).

        Returns
        -------
        Surface
            Surface with points sampled from the parametric function.

        Examples
        --------
        >>> def plane(u, v):
        ...     return torch.stack([u, v, torch.zeros_like(u)])
        >>> u = torch.linspace(0, 1, 10)
        >>> v = torch.linspace(0, 1, 10)
        >>> surface = Surface.from_parametric(plane, u, v)
        """
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.zeros(
            U.shape[0], U.shape[1], 3, dtype=U.dtype, device=U.device
        )
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                points[i, j] = func(U[i, j], V[i, j])
        return Surface(points=points)

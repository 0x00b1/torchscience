"""Quadrature rules for reference finite elements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from torchscience.quadrature import GaussLegendre

if TYPE_CHECKING:
    pass

# Triangle quadrature rules - symmetric rules for reference triangle with
# vertices (0,0), (1,0), (0,1). Area = 0.5
# Points are given in barycentric form (L1, L2, L3) where
# x = L1*v1 + L2*v2 + L3*v3 and L1 + L2 + L3 = 1
# For our reference: v1=(0,0), v2=(1,0), v3=(0,1)
# So x = L2, y = L3 (L1 = 1 - L2 - L3)
# Weights are scaled to integrate over reference triangle (area = 0.5)

# Data format: (order, n_points, [(L1, L2, L3, weight), ...])
# Weights are normalized to sum to 1, then scaled by triangle area
_TRIANGLE_RULES: dict[
    int, tuple[int, list[tuple[float, float, float, float]]]
] = {
    # Order 1: Centroid rule (1 point)
    1: (
        1,
        [
            (1 / 3, 1 / 3, 1 / 3, 1.0),
        ],
    ),
    # Order 2: Midpoint rule (3 points)
    2: (
        3,
        [
            (0.5, 0.5, 0.0, 1 / 3),
            (0.0, 0.5, 0.5, 1 / 3),
            (0.5, 0.0, 0.5, 1 / 3),
        ],
    ),
    # Order 3: 4 points (Hammer-Stroud)
    3: (
        4,
        [
            (1 / 3, 1 / 3, 1 / 3, -27 / 48),
            (0.6, 0.2, 0.2, 25 / 48),
            (0.2, 0.6, 0.2, 25 / 48),
            (0.2, 0.2, 0.6, 25 / 48),
        ],
    ),
    # Order 4: 6 points
    4: (
        6,
        [
            (
                0.816847572980459,
                0.091576213509771,
                0.091576213509771,
                0.109951743655322,
            ),
            (
                0.091576213509771,
                0.816847572980459,
                0.091576213509771,
                0.109951743655322,
            ),
            (
                0.091576213509771,
                0.091576213509771,
                0.816847572980459,
                0.109951743655322,
            ),
            (
                0.108103018168070,
                0.445948490915965,
                0.445948490915965,
                0.223381589678011,
            ),
            (
                0.445948490915965,
                0.108103018168070,
                0.445948490915965,
                0.223381589678011,
            ),
            (
                0.445948490915965,
                0.445948490915965,
                0.108103018168070,
                0.223381589678011,
            ),
        ],
    ),
    # Order 5: 7 points
    5: (
        7,
        [
            (1 / 3, 1 / 3, 1 / 3, 0.225),
            (
                0.797426985353087,
                0.101286507323456,
                0.101286507323456,
                0.125939180544827,
            ),
            (
                0.101286507323456,
                0.797426985353087,
                0.101286507323456,
                0.125939180544827,
            ),
            (
                0.101286507323456,
                0.101286507323456,
                0.797426985353087,
                0.125939180544827,
            ),
            (
                0.059715871789770,
                0.470142064105115,
                0.470142064105115,
                0.132394152788506,
            ),
            (
                0.470142064105115,
                0.059715871789770,
                0.470142064105115,
                0.132394152788506,
            ),
            (
                0.470142064105115,
                0.470142064105115,
                0.059715871789770,
                0.132394152788506,
            ),
        ],
    ),
}

# Tetrahedron quadrature rules - symmetric rules for reference tetrahedron with
# vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1). Volume = 1/6
# Points given in barycentric form (L1, L2, L3, L4)
_TETRAHEDRON_RULES: dict[
    int, tuple[int, list[tuple[float, float, float, float, float]]]
] = {
    # Order 1: Centroid rule (1 point)
    1: (
        1,
        [
            (0.25, 0.25, 0.25, 0.25, 1.0),
        ],
    ),
    # Order 2: 4 points (vertices of inner tetrahedron)
    2: (
        4,
        [
            (
                0.5854101966249685,
                0.1381966011250105,
                0.1381966011250105,
                0.1381966011250105,
                0.25,
            ),
            (
                0.1381966011250105,
                0.5854101966249685,
                0.1381966011250105,
                0.1381966011250105,
                0.25,
            ),
            (
                0.1381966011250105,
                0.1381966011250105,
                0.5854101966249685,
                0.1381966011250105,
                0.25,
            ),
            (
                0.1381966011250105,
                0.1381966011250105,
                0.1381966011250105,
                0.5854101966249685,
                0.25,
            ),
        ],
    ),
    # Order 3: 5 points (Keast)
    3: (
        5,
        [
            (0.25, 0.25, 0.25, 0.25, -0.8),
            (0.5, 1 / 6, 1 / 6, 1 / 6, 0.45),
            (1 / 6, 0.5, 1 / 6, 1 / 6, 0.45),
            (1 / 6, 1 / 6, 0.5, 1 / 6, 0.45),
            (1 / 6, 1 / 6, 1 / 6, 0.5, 0.45),
        ],
    ),
}


def _line_quadrature(
    order: int,
    dtype: torch.dtype,
    device: torch.device | None,
) -> tuple[Tensor, Tensor]:
    """Quadrature for line element [0, 1].

    Uses Gauss-Legendre mapped from [-1, 1] to [0, 1].
    """
    # Gauss-Legendre with n points is exact for polynomials of degree 2n-1
    # So for order k, we need n >= (k+1)/2
    n_points = (order + 2) // 2  # Ceiling division

    rule = GaussLegendre(n_points)
    nodes, weights = rule.nodes_and_weights(
        a=0.0, b=1.0, dtype=dtype, device=device
    )

    # Shape: (n_points,) -> (n_points, 1)
    points = nodes.unsqueeze(-1)
    return points, weights


def _triangle_quadrature(
    order: int,
    dtype: torch.dtype,
    device: torch.device | None,
) -> tuple[Tensor, Tensor]:
    """Quadrature for triangle with vertices (0,0), (1,0), (0,1).

    Area of reference triangle = 0.5
    """
    if order not in _TRIANGLE_RULES:
        available = sorted(_TRIANGLE_RULES.keys())
        raise ValueError(
            f"Triangle quadrature order {order} not available. "
            f"Available orders: {available}"
        )

    n_points, rule_data = _TRIANGLE_RULES[order]

    # Convert barycentric to Cartesian
    # x = L2, y = L3 for reference triangle (0,0), (1,0), (0,1)
    points_list = []
    weights_list = []

    for entry in rule_data:
        L1, L2, L3, w = entry
        x = L2
        y = L3
        points_list.append([x, y])
        weights_list.append(w)

    points = torch.tensor(points_list, dtype=dtype, device=device)
    weights = torch.tensor(weights_list, dtype=dtype, device=device)

    # Scale weights by triangle area (0.5)
    weights = weights * 0.5

    return points, weights


def _quad_quadrature(
    order: int,
    dtype: torch.dtype,
    device: torch.device | None,
) -> tuple[Tensor, Tensor]:
    """Quadrature for quad element [-1, 1] x [-1, 1].

    Uses tensor product of 1D Gauss-Legendre rules.
    Area = 4.
    """
    # For 2D tensor product, n^2 points give order 2n-1 in each direction
    n_points_1d = (order + 2) // 2

    rule = GaussLegendre(n_points_1d)
    nodes_1d, weights_1d = rule.nodes_and_weights(
        dtype=dtype, device=device
    )  # on [-1, 1]

    # Tensor product
    n = n_points_1d
    # Create grid
    x = nodes_1d.unsqueeze(1).expand(n, n).reshape(-1)
    y = nodes_1d.unsqueeze(0).expand(n, n).reshape(-1)
    points = torch.stack([x, y], dim=-1)

    # Product weights
    w_x = weights_1d.unsqueeze(1).expand(n, n).reshape(-1)
    w_y = weights_1d.unsqueeze(0).expand(n, n).reshape(-1)
    weights = w_x * w_y

    return points, weights


def _tetrahedron_quadrature(
    order: int,
    dtype: torch.dtype,
    device: torch.device | None,
) -> tuple[Tensor, Tensor]:
    """Quadrature for tetrahedron with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1).

    Volume of reference tetrahedron = 1/6.
    """
    if order not in _TETRAHEDRON_RULES:
        available = sorted(_TETRAHEDRON_RULES.keys())
        raise ValueError(
            f"Tetrahedron quadrature order {order} not available. "
            f"Available orders: {available}"
        )

    n_points, rule_data = _TETRAHEDRON_RULES[order]

    # Convert barycentric to Cartesian
    # x = L2, y = L3, z = L4 for reference tetrahedron
    points_list = []
    weights_list = []

    for entry in rule_data:
        L1, L2, L3, L4, w = entry
        x = L2
        y = L3
        z = L4
        points_list.append([x, y, z])
        weights_list.append(w)

    points = torch.tensor(points_list, dtype=dtype, device=device)
    weights = torch.tensor(weights_list, dtype=dtype, device=device)

    # Scale weights by tetrahedron volume (1/6)
    weights = weights * (1.0 / 6.0)

    return points, weights


def _hexahedron_quadrature(
    order: int,
    dtype: torch.dtype,
    device: torch.device | None,
) -> tuple[Tensor, Tensor]:
    """Quadrature for hexahedron [-1, 1]^3.

    Uses tensor product of 1D Gauss-Legendre rules.
    Volume = 8.
    """
    n_points_1d = (order + 2) // 2

    rule = GaussLegendre(n_points_1d)
    nodes_1d, weights_1d = rule.nodes_and_weights(dtype=dtype, device=device)

    n = n_points_1d
    # Create 3D grid
    x = nodes_1d.reshape(n, 1, 1).expand(n, n, n).reshape(-1)
    y = nodes_1d.reshape(1, n, 1).expand(n, n, n).reshape(-1)
    z = nodes_1d.reshape(1, 1, n).expand(n, n, n).reshape(-1)
    points = torch.stack([x, y, z], dim=-1)

    # Product weights
    w_x = weights_1d.reshape(n, 1, 1).expand(n, n, n).reshape(-1)
    w_y = weights_1d.reshape(1, n, 1).expand(n, n, n).reshape(-1)
    w_z = weights_1d.reshape(1, 1, n).expand(n, n, n).reshape(-1)
    weights = w_x * w_y * w_z

    return points, weights


_ELEMENT_QUADRATURE = {
    "line": _line_quadrature,
    "triangle": _triangle_quadrature,
    "quad": _quad_quadrature,
    "tetrahedron": _tetrahedron_quadrature,
    "hexahedron": _hexahedron_quadrature,
}


def quadrature_points(
    element_type: str,
    order: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Get quadrature points and weights for a reference element.

    Parameters
    ----------
    element_type : str
        Element type: "line", "triangle", "quad", "tetrahedron", or "hexahedron".
    order : int
        Polynomial order to integrate exactly.
    dtype : torch.dtype, optional
        Output dtype. Default is float64.
    device : torch.device, optional
        Output device.

    Returns
    -------
    points : Tensor
        Quadrature points, shape (num_points, dim).
    weights : Tensor
        Quadrature weights, shape (num_points,).

    Notes
    -----
    Reference element definitions:
        - line: [0, 1], length = 1
        - triangle: vertices (0,0), (1,0), (0,1), area = 0.5
        - quad: [-1,1] x [-1,1], area = 4
        - tetrahedron: vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1), volume = 1/6
        - hexahedron: [-1,1]^3, volume = 8

    Examples
    --------
    >>> points, weights = quadrature_points("triangle", order=2)
    >>> points.shape
    torch.Size([3, 2])
    >>> weights.sum()  # Should equal triangle area = 0.5
    tensor(0.5000, dtype=torch.float64)
    """
    if dtype is None:
        dtype = torch.float64

    element_type_lower = element_type.lower()

    if element_type_lower not in _ELEMENT_QUADRATURE:
        available = list(_ELEMENT_QUADRATURE.keys())
        raise ValueError(
            f"Unknown element type '{element_type}'. Available types: {available}"
        )

    return _ELEMENT_QUADRATURE[element_type_lower](order, dtype, device)

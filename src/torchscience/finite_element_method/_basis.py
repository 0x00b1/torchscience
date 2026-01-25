"""Lagrange basis functions for reference finite elements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    pass


def _num_basis_functions(element_type: str, order: int) -> int:
    """Return the number of basis functions for given element type and order."""
    if element_type == "line":
        return order + 1
    elif element_type == "triangle":
        return (order + 1) * (order + 2) // 2
    elif element_type == "quad":
        return (order + 1) ** 2
    elif element_type == "tetrahedron":
        return (order + 1) * (order + 2) * (order + 3) // 6
    elif element_type == "hexahedron":
        return (order + 1) ** 3
    else:
        raise ValueError(f"Unknown element type '{element_type}'")


def _lagrange_1d_nodes(
    order: int, device: torch.device | None, dtype: torch.dtype
) -> Tensor:
    """Return 1D Lagrange interpolation nodes on [0, 1]."""
    if order == 0:
        return torch.tensor([0.5], device=device, dtype=dtype)
    return torch.linspace(0, 1, order + 1, device=device, dtype=dtype)


def _lagrange_1d_nodes_symmetric(
    order: int, device: torch.device | None, dtype: torch.dtype
) -> Tensor:
    """Return 1D Lagrange interpolation nodes on [-1, 1]."""
    if order == 0:
        return torch.tensor([0.0], device=device, dtype=dtype)
    return torch.linspace(-1, 1, order + 1, device=device, dtype=dtype)


def _lagrange_1d_basis(x: Tensor, nodes: Tensor) -> Tensor:
    """Evaluate 1D Lagrange basis functions at points x.

    Parameters
    ----------
    x : Tensor
        Evaluation points, shape (num_points,).
    nodes : Tensor
        Lagrange interpolation nodes, shape (num_nodes,).

    Returns
    -------
    Tensor
        Basis function values, shape (num_points, num_nodes).
    """
    # x: (num_points,)
    # nodes: (num_nodes,)
    num_points = x.shape[0]
    num_nodes = nodes.shape[0]

    # Compute L_i(x) = prod_{j != i} (x - nodes[j]) / (nodes[i] - nodes[j])
    # Use vectorized form

    # x_expanded: (num_points, 1, 1)
    # nodes_expanded: (1, num_nodes, 1)
    x_expanded = x.unsqueeze(-1).unsqueeze(-1)  # (num_points, 1, 1)
    nodes_i = nodes.unsqueeze(0).unsqueeze(-1)  # (1, num_nodes, 1)
    nodes_j = nodes.unsqueeze(0).unsqueeze(1)  # (1, 1, num_nodes)

    # (x - nodes_j): (num_points, 1, num_nodes)
    numerator = x_expanded - nodes_j  # (num_points, 1, num_nodes)

    # (nodes_i - nodes_j): (1, num_nodes, num_nodes)
    denominator = nodes_i - nodes_j  # (1, num_nodes, num_nodes)

    # Avoid division by zero on diagonal
    # Set diagonal of denominator to 1 (will be masked anyway)
    mask = torch.eye(num_nodes, device=x.device, dtype=torch.bool)
    denominator = denominator.masked_fill(mask.unsqueeze(0), 1.0)
    numerator = numerator.expand(num_points, num_nodes, num_nodes)
    numerator = numerator.masked_fill(mask.unsqueeze(0), 1.0)

    # Compute product over j
    ratio = numerator / denominator  # (num_points, num_nodes, num_nodes)
    basis = ratio.prod(dim=-1)  # (num_points, num_nodes)

    return basis


def _lagrange_1d_basis_gradient(x: Tensor, nodes: Tensor) -> Tensor:
    """Evaluate gradients of 1D Lagrange basis functions at points x.

    Parameters
    ----------
    x : Tensor
        Evaluation points, shape (num_points,).
    nodes : Tensor
        Lagrange interpolation nodes, shape (num_nodes,).

    Returns
    -------
    Tensor
        Basis gradients, shape (num_points, num_nodes).
    """
    num_points = x.shape[0]
    num_nodes = nodes.shape[0]

    # dL_i/dx = L_i(x) * sum_{j != i} 1 / (x - nodes[j])
    # But this has issues at nodes. Use a more stable formula:
    # dL_i/dx = sum_{k != i} prod_{j != i, j != k} (x - nodes[j]) / prod_{j != i} (nodes[i] - nodes[j])

    # Compute the denominator product for each basis function
    # prod_{j != i} (nodes[i] - nodes[j])
    nodes_i = nodes.unsqueeze(1)  # (num_nodes, 1)
    nodes_j = nodes.unsqueeze(0)  # (1, num_nodes)
    denom_all = nodes_i - nodes_j  # (num_nodes, num_nodes)

    # Mask diagonal
    mask = torch.eye(num_nodes, device=x.device, dtype=torch.bool)
    denom_all = denom_all.masked_fill(mask, 1.0)
    denom_prod = denom_all.prod(dim=1)  # (num_nodes,)

    # For the numerator, we need:
    # sum_{k != i} prod_{j != i, j != k} (x - nodes[j])
    # This equals sum over k of: prod over j != i,k of (x - xj)

    grad = torch.zeros(num_points, num_nodes, device=x.device, dtype=x.dtype)

    for i in range(num_nodes):
        # Indices j != i
        other_indices = [j for j in range(num_nodes) if j != i]

        for k_idx, k in enumerate(other_indices):
            # prod over j != i, j != k
            term = torch.ones(num_points, device=x.device, dtype=x.dtype)
            for j in other_indices:
                if j != k:
                    term = term * (x - nodes[j])
            grad[:, i] += term

        grad[:, i] /= denom_prod[i]

    return grad


# =============================================================================
# Line element basis functions
# =============================================================================


def _line_lagrange_basis(points: Tensor, order: int) -> Tensor:
    """Lagrange basis for line element on [0, 1]."""
    x = points[:, 0]  # (num_points,)
    nodes = _lagrange_1d_nodes(order, points.device, points.dtype)
    return _lagrange_1d_basis(x, nodes)


def _line_lagrange_basis_gradient(points: Tensor, order: int) -> Tensor:
    """Gradient of Lagrange basis for line element."""
    x = points[:, 0]
    nodes = _lagrange_1d_nodes(order, points.device, points.dtype)
    grad = _lagrange_1d_basis_gradient(x, nodes)  # (num_points, num_nodes)
    return grad.unsqueeze(-1)  # (num_points, num_nodes, 1)


# =============================================================================
# Triangle element basis functions
# =============================================================================


def _triangle_p1_basis(points: Tensor) -> Tensor:
    """P1 triangle basis: N1 = 1-x-y, N2 = x, N3 = y."""
    x = points[:, 0]
    y = points[:, 1]
    N1 = 1 - x - y
    N2 = x
    N3 = y
    return torch.stack([N1, N2, N3], dim=-1)


def _triangle_p1_basis_gradient(points: Tensor) -> Tensor:
    """Gradient of P1 triangle basis."""
    num_points = points.shape[0]
    device = points.device
    dtype = points.dtype

    # Constant gradients
    grad = torch.zeros(num_points, 3, 2, device=device, dtype=dtype)
    grad[:, 0, 0] = -1  # dN1/dx
    grad[:, 0, 1] = -1  # dN1/dy
    grad[:, 1, 0] = 1  # dN2/dx
    grad[:, 1, 1] = 0  # dN2/dy
    grad[:, 2, 0] = 0  # dN3/dx
    grad[:, 2, 1] = 1  # dN3/dy
    return grad


def _triangle_p2_basis(points: Tensor) -> Tensor:
    """P2 triangle basis (6 nodes)."""
    x = points[:, 0]
    y = points[:, 1]
    L1 = 1 - x - y  # barycentric coord
    L2 = x
    L3 = y

    # Ordering: 3 vertices, then 3 edge midpoints
    N1 = L1 * (2 * L1 - 1)  # vertex (0, 0)
    N2 = L2 * (2 * L2 - 1)  # vertex (1, 0)
    N3 = L3 * (2 * L3 - 1)  # vertex (0, 1)
    N4 = 4 * L1 * L2  # midpoint of edge 1-2
    N5 = 4 * L2 * L3  # midpoint of edge 2-3
    N6 = 4 * L3 * L1  # midpoint of edge 3-1

    return torch.stack([N1, N2, N3, N4, N5, N6], dim=-1)


def _triangle_p2_basis_gradient(points: Tensor) -> Tensor:
    """Gradient of P2 triangle basis."""
    x = points[:, 0]
    y = points[:, 1]
    L1 = 1 - x - y
    L2 = x
    L3 = y

    num_points = points.shape[0]
    device = points.device
    dtype = points.dtype

    grad = torch.zeros(num_points, 6, 2, device=device, dtype=dtype)

    # dL1/dx = -1, dL1/dy = -1
    # dL2/dx = 1,  dL2/dy = 0
    # dL3/dx = 0,  dL3/dy = 1

    # N1 = L1 * (2*L1 - 1)
    # dN1/dx = dL1/dx * (2*L1 - 1) + L1 * 2 * dL1/dx = -1 * (2*L1 - 1) + L1 * (-2) = -(2*L1 - 1) - 2*L1 = -4*L1 + 1
    # dN1/dy = -1 * (2*L1 - 1) - 2*L1 = -4*L1 + 1
    grad[:, 0, 0] = -4 * L1 + 1
    grad[:, 0, 1] = -4 * L1 + 1

    # N2 = L2 * (2*L2 - 1)
    # dN2/dx = 1 * (2*L2 - 1) + L2 * 2 = 4*L2 - 1
    # dN2/dy = 0
    grad[:, 1, 0] = 4 * L2 - 1
    grad[:, 1, 1] = 0

    # N3 = L3 * (2*L3 - 1)
    # dN3/dx = 0
    # dN3/dy = 1 * (2*L3 - 1) + L3 * 2 = 4*L3 - 1
    grad[:, 2, 0] = 0
    grad[:, 2, 1] = 4 * L3 - 1

    # N4 = 4 * L1 * L2
    # dN4/dx = 4 * (dL1/dx * L2 + L1 * dL2/dx) = 4 * (-L2 + L1) = 4 * (L1 - L2)
    # dN4/dy = 4 * (-L2 + 0) = -4 * L2
    grad[:, 3, 0] = 4 * (L1 - L2)
    grad[:, 3, 1] = -4 * L2

    # N5 = 4 * L2 * L3
    # dN5/dx = 4 * (dL2/dx * L3 + L2 * dL3/dx) = 4 * (L3 + 0) = 4 * L3
    # dN5/dy = 4 * (0 + L2) = 4 * L2
    grad[:, 4, 0] = 4 * L3
    grad[:, 4, 1] = 4 * L2

    # N6 = 4 * L3 * L1
    # dN6/dx = 4 * (dL3/dx * L1 + L3 * dL1/dx) = 4 * (0 - L3) = -4 * L3
    # dN6/dy = 4 * (L1 + L3 * (-1)) = 4 * (L1 - L3)
    grad[:, 5, 0] = -4 * L3
    grad[:, 5, 1] = 4 * (L1 - L3)

    return grad


def _triangle_lagrange_basis(points: Tensor, order: int) -> Tensor:
    """Lagrange basis for triangle element."""
    if order == 1:
        return _triangle_p1_basis(points)
    elif order == 2:
        return _triangle_p2_basis(points)
    else:
        raise ValueError(
            f"Triangle basis order {order} not supported. Use order 1 or 2."
        )


def _triangle_lagrange_basis_gradient(points: Tensor, order: int) -> Tensor:
    """Gradient of Lagrange basis for triangle element."""
    if order == 1:
        return _triangle_p1_basis_gradient(points)
    elif order == 2:
        return _triangle_p2_basis_gradient(points)
    else:
        raise ValueError(
            f"Triangle basis order {order} not supported. Use order 1 or 2."
        )


# =============================================================================
# Quad element basis functions (tensor product on [-1, 1]^2)
# =============================================================================


def _quad_lagrange_basis(points: Tensor, order: int) -> Tensor:
    """Lagrange basis for quad element on [-1, 1]^2 using tensor product."""
    x = points[:, 0]
    y = points[:, 1]
    nodes = _lagrange_1d_nodes_symmetric(order, points.device, points.dtype)

    # Evaluate 1D basis functions
    phi_x = _lagrange_1d_basis(x, nodes)  # (num_points, n_nodes_1d)
    phi_y = _lagrange_1d_basis(y, nodes)  # (num_points, n_nodes_1d)

    # Tensor product: N_{ij}(x, y) = phi_i(x) * phi_j(y)
    # Ordering: standard tensor product ordering
    n = order + 1
    num_points = points.shape[0]
    num_basis = n * n

    basis = torch.zeros(
        num_points, num_basis, device=points.device, dtype=points.dtype
    )

    # Standard ordering for quads: (i, j) -> j * n + i (or row-major with y then x)
    # Use a consistent ordering: loop over y first, then x
    for j in range(n):
        for i in range(n):
            idx = j * n + i
            basis[:, idx] = phi_x[:, i] * phi_y[:, j]

    return basis


def _quad_lagrange_basis_gradient(points: Tensor, order: int) -> Tensor:
    """Gradient of Lagrange basis for quad element."""
    x = points[:, 0]
    y = points[:, 1]
    nodes = _lagrange_1d_nodes_symmetric(order, points.device, points.dtype)

    phi_x = _lagrange_1d_basis(x, nodes)
    phi_y = _lagrange_1d_basis(y, nodes)
    dphi_x = _lagrange_1d_basis_gradient(x, nodes)
    dphi_y = _lagrange_1d_basis_gradient(y, nodes)

    n = order + 1
    num_points = points.shape[0]
    num_basis = n * n

    grad = torch.zeros(
        num_points, num_basis, 2, device=points.device, dtype=points.dtype
    )

    for j in range(n):
        for i in range(n):
            idx = j * n + i
            grad[:, idx, 0] = dphi_x[:, i] * phi_y[:, j]
            grad[:, idx, 1] = phi_x[:, i] * dphi_y[:, j]

    return grad


# =============================================================================
# Tetrahedron element basis functions
# =============================================================================


def _tetrahedron_p1_basis(points: Tensor) -> Tensor:
    """P1 tetrahedron basis: N1 = 1-x-y-z, N2 = x, N3 = y, N4 = z."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    N1 = 1 - x - y - z
    N2 = x
    N3 = y
    N4 = z
    return torch.stack([N1, N2, N3, N4], dim=-1)


def _tetrahedron_p1_basis_gradient(points: Tensor) -> Tensor:
    """Gradient of P1 tetrahedron basis."""
    num_points = points.shape[0]
    device = points.device
    dtype = points.dtype

    grad = torch.zeros(num_points, 4, 3, device=device, dtype=dtype)
    grad[:, 0, 0] = -1  # dN1/dx
    grad[:, 0, 1] = -1  # dN1/dy
    grad[:, 0, 2] = -1  # dN1/dz
    grad[:, 1, 0] = 1  # dN2/dx
    grad[:, 2, 1] = 1  # dN3/dy
    grad[:, 3, 2] = 1  # dN4/dz
    return grad


def _tetrahedron_p2_basis(points: Tensor) -> Tensor:
    """P2 tetrahedron basis (10 nodes)."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    L1 = 1 - x - y - z
    L2 = x
    L3 = y
    L4 = z

    # 4 vertex functions
    N1 = L1 * (2 * L1 - 1)
    N2 = L2 * (2 * L2 - 1)
    N3 = L3 * (2 * L3 - 1)
    N4 = L4 * (2 * L4 - 1)

    # 6 edge midpoint functions
    N5 = 4 * L1 * L2  # edge 1-2
    N6 = 4 * L2 * L3  # edge 2-3
    N7 = 4 * L1 * L3  # edge 1-3
    N8 = 4 * L1 * L4  # edge 1-4
    N9 = 4 * L2 * L4  # edge 2-4
    N10 = 4 * L3 * L4  # edge 3-4

    return torch.stack([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10], dim=-1)


def _tetrahedron_p2_basis_gradient(points: Tensor) -> Tensor:
    """Gradient of P2 tetrahedron basis."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    L1 = 1 - x - y - z
    L2 = x
    L3 = y
    L4 = z

    num_points = points.shape[0]
    device = points.device
    dtype = points.dtype

    grad = torch.zeros(num_points, 10, 3, device=device, dtype=dtype)

    # dL1/dx = -1, dL1/dy = -1, dL1/dz = -1
    # dL2/dx = 1,  dL2/dy = 0,  dL2/dz = 0
    # dL3/dx = 0,  dL3/dy = 1,  dL3/dz = 0
    # dL4/dx = 0,  dL4/dy = 0,  dL4/dz = 1

    # N1 = L1 * (2*L1 - 1) -> dN1/dxi = dL1/dxi * (4*L1 - 1)
    grad[:, 0, 0] = -4 * L1 + 1
    grad[:, 0, 1] = -4 * L1 + 1
    grad[:, 0, 2] = -4 * L1 + 1

    # N2 = L2 * (2*L2 - 1)
    grad[:, 1, 0] = 4 * L2 - 1

    # N3 = L3 * (2*L3 - 1)
    grad[:, 2, 1] = 4 * L3 - 1

    # N4 = L4 * (2*L4 - 1)
    grad[:, 3, 2] = 4 * L4 - 1

    # N5 = 4 * L1 * L2
    grad[:, 4, 0] = 4 * (L1 - L2)
    grad[:, 4, 1] = -4 * L2
    grad[:, 4, 2] = -4 * L2

    # N6 = 4 * L2 * L3
    grad[:, 5, 0] = 4 * L3
    grad[:, 5, 1] = 4 * L2
    # Explicitly set missing zero derivative for clarity and safety
    grad[:, 5, 2] = 0  # dN6/dz = 0 (N6 = 4*L2*L3 doesn't depend on z)

    # N7 = 4 * L1 * L3
    grad[:, 6, 0] = -4 * L3
    grad[:, 6, 1] = 4 * (L1 - L3)
    grad[:, 6, 2] = -4 * L3

    # N8 = 4 * L1 * L4
    grad[:, 7, 0] = -4 * L4
    grad[:, 7, 1] = -4 * L4
    grad[:, 7, 2] = 4 * (L1 - L4)

    # N9 = 4 * L2 * L4
    grad[:, 8, 0] = 4 * L4
    # Explicitly set missing zero derivative for clarity and safety
    grad[:, 8, 1] = 0  # dN9/dy = 0 (N9 = 4*L2*L4 doesn't depend on y)
    grad[:, 8, 2] = 4 * L2

    # N10 = 4 * L3 * L4
    grad[:, 9, 1] = 4 * L4
    grad[:, 9, 2] = 4 * L3

    return grad


def _tetrahedron_lagrange_basis(points: Tensor, order: int) -> Tensor:
    """Lagrange basis for tetrahedron element."""
    if order == 1:
        return _tetrahedron_p1_basis(points)
    elif order == 2:
        return _tetrahedron_p2_basis(points)
    else:
        raise ValueError(
            f"Tetrahedron basis order {order} not supported. Use order 1 or 2."
        )


def _tetrahedron_lagrange_basis_gradient(points: Tensor, order: int) -> Tensor:
    """Gradient of Lagrange basis for tetrahedron element."""
    if order == 1:
        return _tetrahedron_p1_basis_gradient(points)
    elif order == 2:
        return _tetrahedron_p2_basis_gradient(points)
    else:
        raise ValueError(
            f"Tetrahedron basis order {order} not supported. Use order 1 or 2."
        )


# =============================================================================
# Hexahedron element basis functions (tensor product on [-1, 1]^3)
# =============================================================================


def _hexahedron_lagrange_basis(points: Tensor, order: int) -> Tensor:
    """Lagrange basis for hexahedron element on [-1, 1]^3 using tensor product."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    nodes = _lagrange_1d_nodes_symmetric(order, points.device, points.dtype)

    phi_x = _lagrange_1d_basis(x, nodes)
    phi_y = _lagrange_1d_basis(y, nodes)
    phi_z = _lagrange_1d_basis(z, nodes)

    n = order + 1
    num_points = points.shape[0]
    num_basis = n**3

    basis = torch.zeros(
        num_points, num_basis, device=points.device, dtype=points.dtype
    )

    # Ordering: z slowest, then y, then x fastest
    for k in range(n):
        for j in range(n):
            for i in range(n):
                idx = k * n * n + j * n + i
                basis[:, idx] = phi_x[:, i] * phi_y[:, j] * phi_z[:, k]

    return basis


def _hexahedron_lagrange_basis_gradient(points: Tensor, order: int) -> Tensor:
    """Gradient of Lagrange basis for hexahedron element."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    nodes = _lagrange_1d_nodes_symmetric(order, points.device, points.dtype)

    phi_x = _lagrange_1d_basis(x, nodes)
    phi_y = _lagrange_1d_basis(y, nodes)
    phi_z = _lagrange_1d_basis(z, nodes)
    dphi_x = _lagrange_1d_basis_gradient(x, nodes)
    dphi_y = _lagrange_1d_basis_gradient(y, nodes)
    dphi_z = _lagrange_1d_basis_gradient(z, nodes)

    n = order + 1
    num_points = points.shape[0]
    num_basis = n**3

    grad = torch.zeros(
        num_points, num_basis, 3, device=points.device, dtype=points.dtype
    )

    for k in range(n):
        for j in range(n):
            for i in range(n):
                idx = k * n * n + j * n + i
                grad[:, idx, 0] = dphi_x[:, i] * phi_y[:, j] * phi_z[:, k]
                grad[:, idx, 1] = phi_x[:, i] * dphi_y[:, j] * phi_z[:, k]
                grad[:, idx, 2] = phi_x[:, i] * phi_y[:, j] * dphi_z[:, k]

    return grad


# =============================================================================
# Dispatcher dictionaries
# =============================================================================

_ELEMENT_BASIS = {
    "line": _line_lagrange_basis,
    "triangle": _triangle_lagrange_basis,
    "quad": _quad_lagrange_basis,
    "tetrahedron": _tetrahedron_lagrange_basis,
    "hexahedron": _hexahedron_lagrange_basis,
}

_ELEMENT_BASIS_GRADIENT = {
    "line": _line_lagrange_basis_gradient,
    "triangle": _triangle_lagrange_basis_gradient,
    "quad": _quad_lagrange_basis_gradient,
    "tetrahedron": _tetrahedron_lagrange_basis_gradient,
    "hexahedron": _hexahedron_lagrange_basis_gradient,
}

_ELEMENT_DIM = {
    "line": 1,
    "triangle": 2,
    "quad": 2,
    "tetrahedron": 3,
    "hexahedron": 3,
}

_MAX_ORDER = {
    "line": 5,
    "triangle": 2,
    "quad": 5,
    "tetrahedron": 2,
    "hexahedron": 5,
}


# =============================================================================
# Public API
# =============================================================================


def lagrange_basis(
    element_type: str,
    order: int,
    points: Tensor,
) -> Tensor:
    """Evaluate Lagrange basis functions at given points.

    Parameters
    ----------
    element_type : str
        Element type: "line", "triangle", "quad", "tetrahedron", "hexahedron".
    order : int
        Polynomial order (1=linear, 2=quadratic, etc.).
    points : Tensor
        Evaluation points, shape (num_points, dim).

    Returns
    -------
    Tensor
        Basis function values, shape (num_points, num_basis_functions).

    Notes
    -----
    Reference element definitions:
        - line: [0, 1]
        - triangle: vertices (0,0), (1,0), (0,1)
        - quad: [-1,1] x [-1,1]
        - tetrahedron: vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        - hexahedron: [-1,1]^3

    Number of basis functions:
        - line: order + 1 (P1: 2, P2: 3)
        - triangle: (order+1)(order+2)/2 (P1: 3, P2: 6)
        - quad: (order+1)^2 (Q1: 4, Q2: 9)
        - tetrahedron: (order+1)(order+2)(order+3)/6 (P1: 4, P2: 10)
        - hexahedron: (order+1)^3 (Q1: 8, Q2: 27)

    Examples
    --------
    >>> # P1 triangle basis at centroid
    >>> points = torch.tensor([[1/3, 1/3]])
    >>> basis = lagrange_basis("triangle", order=1, points=points)
    >>> basis
    tensor([[0.3333, 0.3333, 0.3333]])
    """
    element_type_lower = element_type.lower()

    if element_type_lower not in _ELEMENT_BASIS:
        available = list(_ELEMENT_BASIS.keys())
        raise ValueError(
            f"Unknown element type '{element_type}'. Available types: {available}"
        )

    max_order = _MAX_ORDER[element_type_lower]
    if order < 1 or order > max_order:
        raise ValueError(
            f"Order {order} not supported for {element_type}. Use order 1 to {max_order}."
        )

    expected_dim = _ELEMENT_DIM[element_type_lower]
    if points.shape[1] != expected_dim:
        raise ValueError(
            f"Points dimension {points.shape[1]} does not match element dimension {expected_dim} "
            f"for element type '{element_type}'."
        )

    return _ELEMENT_BASIS[element_type_lower](points, order)


def lagrange_basis_gradient(
    element_type: str,
    order: int,
    points: Tensor,
) -> Tensor:
    """Evaluate gradients of Lagrange basis functions.

    Parameters
    ----------
    element_type : str
        Element type: "line", "triangle", "quad", "tetrahedron", "hexahedron".
    order : int
        Polynomial order (1=linear, 2=quadratic, etc.).
    points : Tensor
        Evaluation points, shape (num_points, dim).

    Returns
    -------
    Tensor
        Basis gradients, shape (num_points, num_basis_functions, dim).
        grad[i, j, k] is the k-th partial derivative of the j-th basis function
        evaluated at the i-th point.

    Examples
    --------
    >>> # P1 triangle gradient (constant)
    >>> points = torch.tensor([[0.3, 0.3]])
    >>> grad = lagrange_basis_gradient("triangle", order=1, points=points)
    >>> grad[0]  # Gradients of 3 basis functions
    tensor([[-1., -1.],  # N1 = 1-x-y: gradient is (-1, -1)
            [ 1.,  0.],  # N2 = x: gradient is (1, 0)
            [ 0.,  1.]]) # N3 = y: gradient is (0, 1)
    """
    element_type_lower = element_type.lower()

    if element_type_lower not in _ELEMENT_BASIS_GRADIENT:
        available = list(_ELEMENT_BASIS_GRADIENT.keys())
        raise ValueError(
            f"Unknown element type '{element_type}'. Available types: {available}"
        )

    max_order = _MAX_ORDER[element_type_lower]
    if order < 1 or order > max_order:
        raise ValueError(
            f"Order {order} not supported for {element_type}. Use order 1 to {max_order}."
        )

    expected_dim = _ELEMENT_DIM[element_type_lower]
    if points.shape[1] != expected_dim:
        raise ValueError(
            f"Points dimension {points.shape[1]} does not match element dimension {expected_dim} "
            f"for element type '{element_type}'."
        )

    return _ELEMENT_BASIS_GRADIENT[element_type_lower](points, order)

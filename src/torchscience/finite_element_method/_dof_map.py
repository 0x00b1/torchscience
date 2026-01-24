"""Degree of freedom mapping for finite element methods."""

from __future__ import annotations

from typing import ClassVar

import torch
from tensordict import tensorclass
from torch import Tensor

from torchscience.geometry.mesh import Mesh


@tensorclass
class DOFMap:
    """Degree of freedom mapping.

    Maps local degrees of freedom within each element to global DOF indices.
    For continuous Galerkin methods, DOFs are shared between adjacent elements.
    For discontinuous Galerkin methods, each element has its own DOFs.

    Attributes
    ----------
    order : int
        Polynomial order of the finite element space.
    dofs_per_element : int
        Number of DOFs per element.
    num_global_dofs : int
        Total number of global DOFs in the mesh.
    local_to_global : Tensor
        Mapping from (element, local_dof) to global DOF index.
        Shape (num_elements, dofs_per_element).
    element_type : str
        Element type: "line", "triangle", "quad", "tetrahedron", "hexahedron".
    """

    # DOFs per element for each element type and order
    _DOFS_PER_ELEMENT: ClassVar[dict] = {
        "line": lambda order: order + 1,
        "triangle": lambda order: (order + 1) * (order + 2) // 2,
        "quad": lambda order: (order + 1) ** 2,
        "tetrahedron": lambda order: (order + 1)
        * (order + 2)
        * (order + 3)
        // 6,
        "hexahedron": lambda order: (order + 1) ** 3,
    }

    # Edges per element for each element type
    _EDGES_PER_ELEMENT: ClassVar[dict] = {
        "line": 0,
        "triangle": 3,
        "quad": 4,
        "tetrahedron": 6,
        "hexahedron": 12,
    }

    # Faces per element for each element type
    _FACES_PER_ELEMENT: ClassVar[dict] = {
        "line": 0,
        "triangle": 0,
        "quad": 0,
        "tetrahedron": 4,
        "hexahedron": 6,
    }

    local_to_global: Tensor
    order: int
    dofs_per_element: int
    num_global_dofs: int
    element_type: str


def dof_map(
    mesh: Mesh,
    order: int,
    continuity: str = "C0",
) -> DOFMap:
    """Create DOF mapping for a mesh.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    order : int
        Polynomial order (1 to 4 supported).
    continuity : str
        "C0" for continuous (Lagrange), "discontinuous" for DG.

    Returns
    -------
    DOFMap
        DOF mapping structure.

    Raises
    ------
    ValueError
        If continuity is not "C0" or "discontinuous", or if order is unsupported.

    Examples
    --------
    >>> import torch
    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> from torchscience.finite_element_method import dof_map
    >>> mesh = rectangle_mesh(2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]])
    >>> dm = dof_map(mesh, order=1)
    >>> dm.num_global_dofs
    9

    """
    # Validate inputs
    if continuity not in ("C0", "discontinuous"):
        raise ValueError(
            f"continuity must be 'C0' or 'discontinuous', got '{continuity}'"
        )
    if order < 1 or order > 4:
        raise ValueError(f"Order must be between 1 and 4, got {order}")

    element_type = mesh.element_type.lower()
    if element_type not in DOFMap._DOFS_PER_ELEMENT:
        raise ValueError(f"Unknown element type: {element_type}")

    dofs_per_element = DOFMap._DOFS_PER_ELEMENT[element_type](order)
    device = mesh.elements.device

    if continuity == "discontinuous":
        local_to_global, num_global_dofs = _build_dg_dof_map(
            mesh, order, dofs_per_element, device
        )
    else:
        local_to_global, num_global_dofs = _build_cg_dof_map(
            mesh, order, dofs_per_element, device
        )

    return DOFMap(
        local_to_global=local_to_global,
        order=order,
        dofs_per_element=dofs_per_element,
        num_global_dofs=num_global_dofs,
        element_type=element_type,
        batch_size=[],
    )


def _build_dg_dof_map(
    mesh: Mesh,
    order: int,
    dofs_per_element: int,
    device: torch.device,
) -> tuple[Tensor, int]:
    """Build DOF map for discontinuous Galerkin (no shared DOFs).

    Each element gets its own set of DOFs numbered consecutively.
    """
    num_elements = mesh.num_elements
    num_global_dofs = num_elements * dofs_per_element

    # Each element gets consecutive DOFs: element i gets [i*dofs, (i+1)*dofs)
    local_to_global = torch.arange(
        num_global_dofs, dtype=torch.int64, device=device
    ).reshape(num_elements, dofs_per_element)

    return local_to_global, num_global_dofs


def _build_cg_dof_map(
    mesh: Mesh,
    order: int,
    dofs_per_element: int,
    device: torch.device,
) -> tuple[Tensor, int]:
    """Build DOF map for continuous Galerkin (shared DOFs at element interfaces).

    For order 1 (P1/Q1): DOFs are at vertices only
    For order >= 2: DOFs at vertices, edges, faces, and interior
    """
    element_type = mesh.element_type.lower()
    num_elements = mesh.num_elements
    num_vertices = mesh.num_vertices

    if order == 1:
        # For P1/Q1, DOFs are at vertices, local_to_global = mesh.elements
        local_to_global = mesh.elements.clone()
        num_global_dofs = num_vertices
        return local_to_global, num_global_dofs

    # For higher orders, we need to handle vertex, edge, face, and interior DOFs
    if element_type == "triangle":
        return _build_triangle_cg_dof_map(
            mesh, order, dofs_per_element, device
        )
    elif element_type == "quad":
        return _build_quad_cg_dof_map(mesh, order, dofs_per_element, device)
    else:
        raise ValueError(f"Higher order CG not implemented for {element_type}")


def _build_triangle_cg_dof_map(
    mesh: Mesh,
    order: int,
    dofs_per_element: int,
    device: torch.device,
) -> tuple[Tensor, int]:
    """Build DOF map for triangle elements with order >= 2.

    DOF numbering for P2 triangle:
    - Local nodes 0, 1, 2 are at vertices
    - Local nodes 3, 4, 5 are at edge midpoints:
      - Node 3: edge between vertices 0-1
      - Node 4: edge between vertices 1-2
      - Node 5: edge between vertices 2-0

    Global DOF numbering:
    - First num_vertices DOFs are vertex DOFs (same as vertex indices)
    - Then edge DOFs (order-1 DOFs per edge)
    - Then face DOFs (for order >= 3)
    """
    num_elements = mesh.num_elements
    num_vertices = mesh.num_vertices
    elements = mesh.elements  # (num_elements, 3)

    # Define local edge connectivity: (v0, v1) for each local edge
    # Edge ordering: edge 0 = (0,1), edge 1 = (1,2), edge 2 = (2,0)
    local_edges = torch.tensor(
        [[0, 1], [1, 2], [2, 0]], dtype=torch.int64, device=device
    )

    # Build global edge map: find unique edges and assign global indices
    # For each element, create edges with vertices sorted
    element_edges = elements[:, local_edges]  # (num_elements, 3, 2)
    element_edges_sorted = torch.sort(element_edges, dim=-1).values
    element_edges_flat = element_edges_sorted.reshape(
        -1, 2
    )  # (num_elements*3, 2)

    # Create edge keys for uniqueness
    edge_keys = (
        element_edges_flat[:, 0] * (num_vertices + 1)
        + element_edges_flat[:, 1]
    )
    unique_edge_keys, edge_inverse = torch.unique(
        edge_keys, return_inverse=True
    )
    num_edges = unique_edge_keys.shape[0]

    # edge_inverse maps each (element, local_edge) to a global edge index
    edge_global_indices = edge_inverse.reshape(num_elements, 3)

    # Number of DOFs per edge (interior points along edge)
    edge_dofs_per_edge = order - 1

    # For order=2: 1 DOF per edge (midpoint)
    # For order=3: 2 DOFs per edge, etc.

    # Build local_to_global
    local_to_global = torch.zeros(
        (num_elements, dofs_per_element), dtype=torch.int64, device=device
    )

    # Vertex DOFs: first 3 local DOFs map to vertices
    local_to_global[:, :3] = elements

    if order >= 2:
        # Edge DOFs
        # Global edge DOF index = num_vertices + edge_index * edge_dofs_per_edge + k
        for local_edge in range(3):
            global_edge = edge_global_indices[:, local_edge]
            for k in range(edge_dofs_per_edge):
                local_dof = 3 + local_edge * edge_dofs_per_edge + k
                global_dof = (
                    num_vertices + global_edge * edge_dofs_per_edge + k
                )
                local_to_global[:, local_dof] = global_dof

    num_global_dofs = num_vertices + num_edges * edge_dofs_per_edge

    if order >= 3:
        # Face (interior) DOFs
        # Number of interior DOFs per triangle: (order-1)(order-2)/2
        face_dofs_per_face = (order - 1) * (order - 2) // 2
        if face_dofs_per_face > 0:
            # Face DOFs are unique to each element (no sharing)
            face_dof_start = 3 + 3 * edge_dofs_per_edge  # local index
            for k in range(face_dofs_per_face):
                local_dof = face_dof_start + k
                global_dof = (
                    num_global_dofs
                    + torch.arange(
                        num_elements, dtype=torch.int64, device=device
                    )
                    * face_dofs_per_face
                    + k
                )
                local_to_global[:, local_dof] = global_dof
            num_global_dofs += num_elements * face_dofs_per_face

    return local_to_global, num_global_dofs


def _build_quad_cg_dof_map(
    mesh: Mesh,
    order: int,
    dofs_per_element: int,
    device: torch.device,
) -> tuple[Tensor, int]:
    """Build DOF map for quad elements with order >= 2.

    DOF numbering for Q2 quad (9 nodes):
    - Local nodes 0-3 are at vertices (corners)
    - Local nodes 4-7 are at edge midpoints
    - Local node 8 is at center (face DOF)

    Tensor product ordering: idx = j * (order+1) + i
    where i is x-index, j is y-index.

    For Q2:
    - idx 0: (0,0) = vertex 0 (bottom-left)
    - idx 1: (1,0) = edge 0-1 midpoint
    - idx 2: (2,0) = vertex 1 (bottom-right)
    - idx 3: (0,1) = edge 3-0 midpoint
    - idx 4: (1,1) = center
    - idx 5: (2,1) = edge 1-2 midpoint
    - idx 6: (0,2) = vertex 3 (top-left)
    - idx 7: (1,2) = edge 2-3 midpoint
    - idx 8: (2,2) = vertex 2 (top-right)

    Note: This assumes mesh.elements has ordering [v0, v1, v2, v3] where:
    - v0 = bottom-left, v1 = bottom-right, v2 = top-right, v3 = top-left
    """
    num_elements = mesh.num_elements
    num_vertices = mesh.num_vertices
    elements = mesh.elements  # (num_elements, 4)
    n = order + 1  # nodes per edge

    # Local edges in quad: (v0,v1), (v1,v2), (v2,v3), (v3,v0)
    # Using mesh element ordering where elements are [v0, v1, v2, v3]
    # But rectangle_mesh uses counter-clockwise: [v00, v10, v11, v01]
    # = [bottom-left, bottom-right, top-right, top-left]
    local_edges = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int64, device=device
    )

    # Build global edge map
    element_edges = elements[:, local_edges]  # (num_elements, 4, 2)
    element_edges_sorted = torch.sort(element_edges, dim=-1).values
    element_edges_flat = element_edges_sorted.reshape(-1, 2)

    edge_keys = (
        element_edges_flat[:, 0] * (num_vertices + 1)
        + element_edges_flat[:, 1]
    )
    unique_edge_keys, edge_inverse = torch.unique(
        edge_keys, return_inverse=True
    )
    num_edges = unique_edge_keys.shape[0]
    edge_global_indices = edge_inverse.reshape(num_elements, 4)

    # DOFs per edge (interior points, not including vertices)
    edge_dofs_per_edge = order - 1

    # Build local_to_global
    local_to_global = torch.zeros(
        (num_elements, dofs_per_element), dtype=torch.int64, device=device
    )

    # For tensor product elements, the local DOF ordering is:
    # idx = j * n + i where n = order + 1
    # Corners: (0,0), (n-1,0), (n-1,n-1), (0,n-1)
    # = indices 0, n-1, n*n-1, n*(n-1)
    corner_indices = [0, n - 1, n * n - 1, n * (n - 1)]
    for local_corner, corner_idx in enumerate(corner_indices):
        local_to_global[:, corner_idx] = elements[:, local_corner]

    if order >= 2:
        # Edge DOFs
        # Edge 0 (bottom): y=0, x=1..n-2 -> local indices 1, 2, ..., n-2
        # Edge 1 (right): x=n-1, y=1..n-2 -> local indices n-1+n, n-1+2n, ...
        # Edge 2 (top): y=n-1, x=n-2..1 -> local indices n*(n-1)+n-2, ..., n*(n-1)+1
        # Edge 3 (left): x=0, y=n-2..1 -> local indices n*(n-2), n*(n-3), ...

        for local_edge in range(4):
            global_edge = edge_global_indices[:, local_edge]

            # Get the two vertices of this edge
            v0_idx = local_edges[local_edge, 0]
            v1_idx = local_edges[local_edge, 1]
            v0_global = elements[:, v0_idx]
            v1_global = elements[:, v1_idx]

            # Determine edge orientation: which way do we traverse?
            # Standard edge direction is v0 -> v1
            # We need to check if the global edge is (v0, v1) or (v1, v0) to ensure
            # consistent DOF numbering

            # Get sorted vertices for each element's edge
            sorted_v = torch.stack(
                [
                    torch.minimum(v0_global, v1_global),
                    torch.maximum(v0_global, v1_global),
                ],
                dim=-1,
            )

            # Check if v0 < v1 (forward) or v0 > v1 (backward)
            is_forward = v0_global < v1_global

            for k in range(edge_dofs_per_edge):
                # Local DOF index depends on edge and position
                if local_edge == 0:  # bottom edge
                    local_dof = 1 + k
                elif local_edge == 1:  # right edge
                    local_dof = (n - 1) + (1 + k) * n
                elif local_edge == 2:  # top edge
                    local_dof = n * (n - 1) + (n - 2 - k)
                elif local_edge == 3:  # left edge
                    local_dof = (n - 2 - k) * n

                # Global DOF index
                # If forward: use k, if backward: use (edge_dofs_per_edge - 1 - k)
                global_k = torch.where(
                    is_forward, k, edge_dofs_per_edge - 1 - k
                )
                global_dof = (
                    num_vertices + global_edge * edge_dofs_per_edge + global_k
                )
                local_to_global[:, local_dof] = global_dof

    num_global_dofs = num_vertices + num_edges * edge_dofs_per_edge

    if order >= 2:
        # Face (interior) DOFs - one per element for Q2
        # Interior DOFs: positions (i, j) where 0 < i < n-1 and 0 < j < n-1
        face_dofs_per_face = (order - 1) ** 2
        if face_dofs_per_face > 0:
            interior_dof_count = 0
            for j in range(1, n - 1):
                for i in range(1, n - 1):
                    local_dof = j * n + i
                    global_dof = (
                        num_global_dofs
                        + torch.arange(
                            num_elements, dtype=torch.int64, device=device
                        )
                        * face_dofs_per_face
                        + interior_dof_count
                    )
                    local_to_global[:, local_dof] = global_dof
                    interior_dof_count += 1
            num_global_dofs += num_elements * face_dofs_per_face

    return local_to_global, num_global_dofs

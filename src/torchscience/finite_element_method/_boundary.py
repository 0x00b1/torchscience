"""Boundary condition utilities for finite element methods."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from torchscience.geometry.mesh import (
    Mesh,
    mesh_boundary_facets,
    mesh_boundary_vertices,
)

from ._dof_map import DOFMap


def boundary_dofs(
    mesh: Mesh,
    dof_map: DOFMap,
    marker: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Identify DOFs on the mesh boundary.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    dof_map : DOFMap
        DOF mapping.
    marker : callable, optional
        Function that takes vertex coordinates (num_vertices, dim) and returns
        boolean mask indicating which vertices to include. If None, all
        boundary vertices are included.

    Returns
    -------
    Tensor
        Indices of boundary DOFs, shape (num_boundary_dofs,).

    Notes
    -----
    For P1 elements, boundary DOFs are just the boundary vertices.
    For higher-order elements, boundary DOFs include vertices and edge DOFs
    on boundary edges.

    Examples
    --------
    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> from torchscience.finite_element_method import dof_map, boundary_dofs
    >>> mesh = rectangle_mesh(3, 3)
    >>> dm = dof_map(mesh, order=1)
    >>> b_dofs = boundary_dofs(mesh, dm)
    >>> b_dofs.shape
    torch.Size([12])

    """
    order = dof_map.order
    device = mesh.vertices.device

    if order == 1:
        # For P1, boundary DOFs are just boundary vertices
        return _boundary_dofs_p1(mesh, marker)
    else:
        # For higher order, include vertex DOFs and edge DOFs on boundary edges
        return _boundary_dofs_higher_order(mesh, dof_map, marker)


def _boundary_dofs_p1(
    mesh: Mesh,
    marker: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Get boundary DOFs for P1 elements.

    For P1, DOFs are located at vertices, so boundary DOFs = boundary vertices.
    """
    # Get all boundary vertices
    boundary_verts = mesh_boundary_vertices(mesh)

    if marker is None:
        # Return all boundary vertices sorted
        return torch.sort(boundary_verts).values

    # Apply marker to filter boundary vertices
    coords = mesh.vertices[boundary_verts]
    mask = marker(coords)
    filtered_verts = boundary_verts[mask]

    return torch.sort(filtered_verts).values


def _boundary_dofs_higher_order(
    mesh: Mesh,
    dof_map: DOFMap,
    marker: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Get boundary DOFs for higher-order elements (P2+).

    For higher-order elements, boundary DOFs include:
    - Vertex DOFs on boundary vertices
    - Edge DOFs on boundary edges
    """
    order = dof_map.order
    num_vertices = mesh.num_vertices

    # Get boundary facets (edges in 2D)
    boundary_facets = mesh_boundary_facets(mesh)  # (num_boundary_edges, 2)

    # Get boundary vertices
    boundary_verts = mesh_boundary_vertices(mesh)

    # Number of interior DOFs per edge
    edge_dofs_per_edge = order - 1

    # Collect all boundary DOFs
    all_boundary_dofs = []

    # Add vertex DOFs (for P2+, vertices still have DOFs)
    if marker is None:
        all_boundary_dofs.append(boundary_verts)
    else:
        coords = mesh.vertices[boundary_verts]
        mask = marker(coords)
        all_boundary_dofs.append(boundary_verts[mask])

    # Add edge DOFs for boundary edges
    if edge_dofs_per_edge > 0:
        # We need to map each boundary edge to its global edge index
        # Recompute the edge indexing the same way as in dof_map
        element_type = mesh.element_type.lower()

        if element_type == "triangle":
            edge_dofs = _get_boundary_edge_dofs_triangle(
                mesh, dof_map, boundary_facets, marker
            )
        elif element_type == "quad":
            edge_dofs = _get_boundary_edge_dofs_quad(
                mesh, dof_map, boundary_facets, marker
            )
        else:
            raise ValueError(
                f"Higher-order boundary DOFs not implemented for {element_type}"
            )

        if len(edge_dofs) > 0:
            all_boundary_dofs.append(edge_dofs)

    # Concatenate and sort
    if len(all_boundary_dofs) == 0:
        return torch.tensor([], dtype=torch.long, device=mesh.vertices.device)

    all_boundary_dofs = torch.cat(all_boundary_dofs)
    return torch.sort(torch.unique(all_boundary_dofs)).values


def _get_boundary_edge_dofs_triangle(
    mesh: Mesh,
    dof_map: DOFMap,
    boundary_facets: Tensor,
    marker: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Get edge DOFs on boundary for triangle meshes."""
    order = dof_map.order
    device = mesh.vertices.device
    num_vertices = mesh.num_vertices
    elements = mesh.elements

    edge_dofs_per_edge = order - 1

    # Define local edge connectivity
    local_edges = torch.tensor(
        [[0, 1], [1, 2], [2, 0]], dtype=torch.int64, device=device
    )

    # Build global edge map (same as in dof_map)
    element_edges = elements[:, local_edges]  # (num_elements, 3, 2)
    element_edges_sorted = torch.sort(element_edges, dim=-1).values
    element_edges_flat = element_edges_sorted.reshape(-1, 2)

    edge_keys = (
        element_edges_flat[:, 0] * (num_vertices + 1)
        + element_edges_flat[:, 1]
    )
    unique_edge_keys, edge_inverse = torch.unique(
        edge_keys, return_inverse=True
    )

    # Process boundary facets
    boundary_facets_sorted = torch.sort(boundary_facets, dim=-1).values
    boundary_edge_keys = (
        boundary_facets_sorted[:, 0] * (num_vertices + 1)
        + boundary_facets_sorted[:, 1]
    )

    # Find which boundary edges exist in the global edge set using tensor ops
    # Use searchsorted to find indices (vectorized lookup)
    sorted_unique_keys, sort_perm = torch.sort(unique_edge_keys)
    search_indices = torch.searchsorted(sorted_unique_keys, boundary_edge_keys)

    # Clamp indices to valid range for safe indexing
    search_indices_clamped = torch.clamp(
        search_indices, 0, len(sorted_unique_keys) - 1
    )

    # Check which boundary edges were actually found
    found_mask = (
        sorted_unique_keys[search_indices_clamped] == boundary_edge_keys
    )

    # Get original global edge indices for found edges
    global_edges = sort_perm[search_indices_clamped]

    # Apply marker filter if provided
    if marker is not None:
        # Compute edge midpoints for all boundary facets
        v0_coords = mesh.vertices[boundary_facets[:, 0]]
        v1_coords = mesh.vertices[boundary_facets[:, 1]]
        edge_midpoints = (v0_coords + v1_coords) / 2
        marker_mask = marker(edge_midpoints)
        found_mask = found_mask & marker_mask

    # Filter to only valid edges
    valid_global_edges = global_edges[found_mask]

    if len(valid_global_edges) == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    # Generate all DOF indices for valid edges using tensor operations
    # For each edge, we need edge_dofs_per_edge DOFs
    num_valid_edges = len(valid_global_edges)
    dof_offsets = torch.arange(
        edge_dofs_per_edge, dtype=torch.long, device=device
    )

    # Broadcast: (num_valid_edges, 1) + (edge_dofs_per_edge,) -> (num_valid_edges, edge_dofs_per_edge)
    edge_dofs = (
        num_vertices
        + valid_global_edges.unsqueeze(1) * edge_dofs_per_edge
        + dof_offsets
    )

    return edge_dofs.flatten()


def _get_boundary_edge_dofs_quad(
    mesh: Mesh,
    dof_map: DOFMap,
    boundary_facets: Tensor,
    marker: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Get edge DOFs on boundary for quad meshes."""
    order = dof_map.order
    device = mesh.vertices.device
    num_vertices = mesh.num_vertices
    elements = mesh.elements

    edge_dofs_per_edge = order - 1

    # Define local edge connectivity for quads
    local_edges = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int64, device=device
    )

    # Build global edge map (same as in dof_map)
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

    # Process boundary facets
    boundary_facets_sorted = torch.sort(boundary_facets, dim=-1).values
    boundary_edge_keys = (
        boundary_facets_sorted[:, 0] * (num_vertices + 1)
        + boundary_facets_sorted[:, 1]
    )

    # Find which boundary edges exist in the global edge set using tensor ops
    # Use searchsorted to find indices (vectorized lookup)
    sorted_unique_keys, sort_perm = torch.sort(unique_edge_keys)
    search_indices = torch.searchsorted(sorted_unique_keys, boundary_edge_keys)

    # Clamp indices to valid range for safe indexing
    search_indices_clamped = torch.clamp(
        search_indices, 0, len(sorted_unique_keys) - 1
    )

    # Check which boundary edges were actually found
    found_mask = (
        sorted_unique_keys[search_indices_clamped] == boundary_edge_keys
    )

    # Get original global edge indices for found edges
    global_edges = sort_perm[search_indices_clamped]

    # Apply marker filter if provided
    if marker is not None:
        # Compute edge midpoints for all boundary facets
        v0_coords = mesh.vertices[boundary_facets[:, 0]]
        v1_coords = mesh.vertices[boundary_facets[:, 1]]
        edge_midpoints = (v0_coords + v1_coords) / 2
        marker_mask = marker(edge_midpoints)
        found_mask = found_mask & marker_mask

    # Filter to only valid edges
    valid_global_edges = global_edges[found_mask]

    if len(valid_global_edges) == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    # Generate all DOF indices for valid edges using tensor operations
    # For each edge, we need edge_dofs_per_edge DOFs
    num_valid_edges = len(valid_global_edges)
    dof_offsets = torch.arange(
        edge_dofs_per_edge, dtype=torch.long, device=device
    )

    # Broadcast: (num_valid_edges, 1) + (edge_dofs_per_edge,) -> (num_valid_edges, edge_dofs_per_edge)
    edge_dofs = (
        num_vertices
        + valid_global_edges.unsqueeze(1) * edge_dofs_per_edge
        + dof_offsets
    )

    return edge_dofs.flatten()


def apply_dirichlet_penalty(
    matrix: Tensor,
    vector: Tensor,
    dofs: Tensor,
    values: Tensor,
    penalty: float = 1e10,
) -> tuple[Tensor, Tensor]:
    """Apply Dirichlet boundary conditions using the penalty method.

    Modifies the system K @ u = f by adding a large penalty to constrained DOFs:
    - K[i,i] += penalty for each constrained DOF i
    - f[i] += penalty * value[i] for each constrained DOF i

    Parameters
    ----------
    matrix : Tensor
        System matrix (sparse CSR), shape (n, n).
    vector : Tensor
        Load vector (dense), shape (n,).
    dofs : Tensor
        DOF indices to constrain, shape (num_bc,).
    values : Tensor
        Prescribed values at constrained DOFs, shape (num_bc,).
    penalty : float, optional
        Penalty factor. Default 1e10.

    Returns
    -------
    matrix : Tensor
        Modified system matrix (sparse CSR).
    vector : Tensor
        Modified load vector.

    Notes
    -----
    The penalty method is simple but introduces numerical error proportional
    to 1/penalty. For most applications, penalty=1e10 provides sufficient
    accuracy while maintaining reasonable conditioning.

    Examples
    --------
    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> from torchscience.finite_element_method import (
    ...     dof_map, boundary_dofs, apply_dirichlet_penalty
    ... )
    >>> mesh = rectangle_mesh(2, 2)
    >>> dm = dof_map(mesh, order=1)
    >>> K = torch.eye(9, dtype=torch.float64).to_sparse_csr()
    >>> f = torch.zeros(9, dtype=torch.float64)
    >>> bc_dofs = boundary_dofs(mesh, dm)
    >>> bc_values = torch.ones(len(bc_dofs), dtype=torch.float64)
    >>> K_mod, f_mod = apply_dirichlet_penalty(K, f, bc_dofs, bc_values)

    """
    # Validate device compatibility
    if matrix.device != vector.device:
        raise ValueError(
            f"matrix and vector must be on the same device, "
            f"got {matrix.device} and {vector.device}"
        )
    if matrix.device != dofs.device:
        raise ValueError(
            f"matrix and dofs must be on the same device, "
            f"got {matrix.device} and {dofs.device}"
        )
    if matrix.device != values.device:
        raise ValueError(
            f"matrix and values must be on the same device, "
            f"got {matrix.device} and {values.device}"
        )

    # Handle empty dofs case - return unchanged matrix and vector
    if dofs.numel() == 0:
        return matrix, vector

    # TODO: For large-scale problems, implement sparse-aware diagonal modification
    # to avoid O(n^2) memory usage. Current implementation converts to dense for
    # simplicity.
    K_dense = matrix.to_dense()
    f_mod = vector.clone()

    # Add penalty to diagonal
    K_dense[dofs, dofs] += penalty

    # Modify RHS
    f_mod[dofs] += penalty * values

    return K_dense.to_sparse_csr(), f_mod

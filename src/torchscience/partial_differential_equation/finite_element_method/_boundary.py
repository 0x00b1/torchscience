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
    >>> from torchscience.partial_differential_equation.finite_element_method import dof_map, boundary_dofs
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
    >>> from torchscience.partial_differential_equation.finite_element_method import (
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


def apply_dirichlet_elimination(
    matrix: Tensor,
    vector: Tensor,
    dofs: Tensor,
    values: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply Dirichlet boundary conditions using the elimination method.

    Modifies the system K @ u = f by eliminating constrained DOFs:
    - For unconstrained DOFs j: f[j] -= K[j,i] * prescribed_value[i]
    - Set row i of K to zero (except diagonal)
    - Set column i of K to zero (except diagonal)
    - Set K[i,i] = 1
    - Set f[i] = prescribed_value[i]

    This maintains symmetry if the original matrix was symmetric.

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

    Returns
    -------
    matrix : Tensor
        Modified system matrix (sparse CSR).
    vector : Tensor
        Modified load vector.

    Notes
    -----
    The elimination method enforces boundary conditions exactly (up to numerical
    precision) without introducing conditioning issues like the penalty method.
    It is more accurate but requires modifying the matrix structure.

    Examples
    --------
    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> from torchscience.partial_differential_equation.finite_element_method import (
    ...     dof_map, boundary_dofs, apply_dirichlet_elimination
    ... )
    >>> mesh = rectangle_mesh(2, 2)
    >>> dm = dof_map(mesh, order=1)
    >>> K = torch.eye(9, dtype=torch.float64).to_sparse_csr()
    >>> f = torch.zeros(9, dtype=torch.float64)
    >>> bc_dofs = boundary_dofs(mesh, dm)
    >>> bc_values = torch.ones(len(bc_dofs), dtype=torch.float64)
    >>> K_mod, f_mod = apply_dirichlet_elimination(K, f, bc_dofs, bc_values)

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

    # TODO: For large-scale problems, implement sparse-aware modification
    # to avoid O(n^2) memory usage. Current implementation converts to dense
    # for simplicity.
    K_dense = matrix.to_dense()
    f_mod = vector.clone()

    # Step 1: Modify RHS for unconstrained DOFs
    # For each unconstrained DOF j: f[j] -= sum_i(K[j,i] * values[i])
    # This accounts for the known values being moved to the RHS
    f_mod = f_mod - K_dense[:, dofs] @ values

    # Step 2: Zero out rows and columns for constrained DOFs
    # Set entire rows to zero
    K_dense[dofs, :] = 0.0

    # Set entire columns to zero
    K_dense[:, dofs] = 0.0

    # Step 3: Set diagonal to 1 for constrained DOFs
    K_dense[dofs, dofs] = 1.0

    # Step 4: Set RHS for constrained DOFs to prescribed values
    f_mod[dofs] = values

    return K_dense.to_sparse_csr(), f_mod


def apply_neumann(
    vector: Tensor,
    mesh: Mesh,
    dof_map: DOFMap,
    boundary_facets: Tensor,
    flux: Tensor | Callable[[Tensor], Tensor],
) -> Tensor:
    """Apply Neumann boundary conditions.

    Neumann boundary conditions prescribe the normal derivative (flux) on the
    boundary. In the weak form, this adds a boundary integral to the right-hand
    side load vector:

        f[i] += integral_over_boundary(g * N_i) ds

    where g is the prescribed flux and N_i is the basis function for DOF i.

    For P1 elements on a 2D mesh, each boundary edge contributes to its two
    endpoint DOFs using 2-point Gauss quadrature for accurate integration.

    Parameters
    ----------
    vector : Tensor
        Load vector (dense), shape (n,).
    mesh : Mesh
        The mesh.
    dof_map : DOFMap
        DOF mapping.
    boundary_facets : Tensor
        Boundary facet (edge) indices to apply Neumann BC, shape (num_facets, 2).
        Each row contains vertex indices of an edge.
    flux : Tensor or callable
        Prescribed flux values. Either:
        - Tensor of shape (num_facets,) with constant flux per facet
        - Callable taking coordinates (num_points, dim) and returning flux values

    Returns
    -------
    Tensor
        Modified load vector.

    Notes
    -----
    For P1 elements, we use 2-point Gauss quadrature on each edge:
    - Quadrature points at xi = +/- 1/sqrt(3) in reference coordinates [-1, 1]
    - Weights w1 = w2 = 1.0
    - Shape functions: N1(xi) = (1-xi)/2, N2(xi) = (1+xi)/2

    The contribution to DOF i from edge with vertices (v1, v2) is:
        f[i] += (L/2) * sum_q(w_q * g(x_q) * N_i(xi_q))

    where L is the edge length and (L/2) is the Jacobian of the mapping from
    reference coordinates [-1, 1] to the physical edge.

    Examples
    --------
    >>> from torchscience.geometry.mesh import rectangle_mesh, mesh_boundary_facets
    >>> from torchscience.partial_differential_equation.finite_element_method import dof_map, apply_neumann
    >>> mesh = rectangle_mesh(2, 2)
    >>> dm = dof_map(mesh, order=1)
    >>> f = torch.zeros(dm.num_global_dofs, dtype=torch.float64)
    >>> boundary_facets = mesh_boundary_facets(mesh)
    >>> flux = torch.ones(len(boundary_facets), dtype=torch.float64)
    >>> f_mod = apply_neumann(f, mesh, dm, boundary_facets, flux)

    """
    # Validate boundary_facets shape
    if boundary_facets.ndim != 2 or boundary_facets.shape[1] != 2:
        raise ValueError(
            f"boundary_facets must have shape (num_facets, 2), "
            f"got {boundary_facets.shape}"
        )

    # Handle empty boundary_facets case
    if boundary_facets.numel() == 0:
        return vector.clone()

    # Validate device compatibility
    if boundary_facets.device != vector.device:
        raise ValueError(
            f"vector and boundary_facets must be on the same device, "
            f"got {vector.device} and {boundary_facets.device}"
        )
    if not callable(flux) and flux.device != vector.device:
        raise ValueError(
            f"vector and flux must be on the same device, "
            f"got {vector.device} and {flux.device}"
        )

    device = vector.device
    dtype = vector.dtype
    num_facets = boundary_facets.shape[0]

    # Clone vector to avoid modifying input
    f_mod = vector.clone()

    # Get vertex coordinates for each edge endpoint
    # boundary_facets: (num_facets, 2) - each row is [v1_idx, v2_idx]
    v1_idx = boundary_facets[:, 0]  # (num_facets,)
    v2_idx = boundary_facets[:, 1]  # (num_facets,)

    v1_coords = mesh.vertices[v1_idx]  # (num_facets, dim)
    v2_coords = mesh.vertices[v2_idx]  # (num_facets, dim)

    # Compute edge lengths
    edge_vectors = v2_coords - v1_coords  # (num_facets, dim)
    edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)  # (num_facets,)

    # 2-point Gauss quadrature on reference interval [-1, 1]
    # Points: xi = +/- 1/sqrt(3)
    # Weights: w = 1.0 (each)
    sqrt3_inv = 1.0 / (3.0**0.5)
    xi_q = torch.tensor([-sqrt3_inv, sqrt3_inv], dtype=dtype, device=device)
    w_q = torch.tensor([1.0, 1.0], dtype=dtype, device=device)

    # Shape functions at quadrature points
    # N1(xi) = (1 - xi) / 2  (value at first vertex)
    # N2(xi) = (1 + xi) / 2  (value at second vertex)
    N1_q = (1.0 - xi_q) / 2.0  # (2,)
    N2_q = (1.0 + xi_q) / 2.0  # (2,)

    # Map quadrature points to physical coordinates
    # x(xi) = v1 + (v2 - v1) * (xi + 1) / 2 = v1 * (1-xi)/2 + v2 * (1+xi)/2
    # For each quadrature point and each facet:
    # quad_coords[q, f, :] = N1(xi_q) * v1_coords[f, :] + N2(xi_q) * v2_coords[f, :]
    # Shape: (2, num_facets, dim)
    quad_coords = (
        N1_q[:, None, None] * v1_coords[None, :, :]
        + N2_q[:, None, None] * v2_coords[None, :, :]
    )

    # Evaluate flux at quadrature points
    if callable(flux):
        # flux is a function: evaluate at all quadrature points
        # Reshape to (2 * num_facets, dim) for the call
        num_quad = xi_q.shape[0]
        quad_coords_flat = quad_coords.reshape(
            -1, mesh.dim
        )  # (2*num_facets, dim)
        flux_values_flat = flux(quad_coords_flat)  # (2*num_facets,)
        flux_at_quad = flux_values_flat.reshape(
            num_quad, num_facets
        )  # (2, num_facets)
    else:
        # flux is a tensor of shape (num_facets,) - constant per facet
        # Broadcast to all quadrature points
        flux_at_quad = flux[None, :].expand(2, -1)  # (2, num_facets)

    # Compute contributions to each endpoint DOF
    # Jacobian of mapping from [-1, 1] to physical edge is L/2
    jacobian = edge_lengths / 2.0  # (num_facets,)

    # Contribution to first vertex DOF:
    # f[v1] += sum_q(w_q * jacobian * g(x_q) * N1(xi_q))
    contrib_v1 = torch.einsum(
        "q,f,qf,q->f", w_q, jacobian, flux_at_quad, N1_q
    )  # (num_facets,)

    # Contribution to second vertex DOF:
    # f[v2] += sum_q(w_q * jacobian * g(x_q) * N2(xi_q))
    contrib_v2 = torch.einsum(
        "q,f,qf,q->f", w_q, jacobian, flux_at_quad, N2_q
    )  # (num_facets,)

    # Accumulate contributions using scatter_add
    f_mod.scatter_add_(0, v1_idx, contrib_v1.to(dtype))
    f_mod.scatter_add_(0, v2_idx, contrib_v2.to(dtype))

    return f_mod

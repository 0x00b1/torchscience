"""Mesh boundary detection operators."""

from __future__ import annotations

import torch
from torch import Tensor

from ._mesh import Mesh


def mesh_boundary_facets(mesh: Mesh) -> Tensor:
    """Detect boundary facets of a mesh.

    A facet is on the boundary if it belongs to only one element.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.

    Returns
    -------
    Tensor
        Boundary facet vertex indices, shape (num_boundary_facets, vertices_per_facet).

    Examples
    --------
    >>> from torchscience.geometry.mesh import rectangle_mesh, mesh_boundary_facets
    >>> mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
    >>> boundary = mesh_boundary_facets(mesh)
    >>> boundary.shape
    torch.Size([8, 2])
    """
    element_type = mesh.element_type
    elements = mesh.elements

    # Get facet definitions for this element type
    facet_local = _get_facet_local_indices(element_type)

    # Extract all facets from all elements
    all_facets = []
    for facet_idx in facet_local:
        facets = elements[:, facet_idx]  # (num_elements, vertices_per_facet)
        all_facets.append(facets)

    all_facets = torch.cat(
        all_facets, dim=0
    )  # (num_elements * facets_per_element, vpf)

    # Sort vertices within each facet for comparison
    sorted_facets, _ = torch.sort(all_facets, dim=1)

    # Find unique facets and their counts
    unique_facets, _, counts = torch.unique(
        sorted_facets, dim=0, return_inverse=True, return_counts=True
    )

    # Boundary facets appear exactly once
    boundary_mask = counts == 1
    boundary_facets = unique_facets[boundary_mask]

    return boundary_facets


def mesh_boundary_vertices(mesh: Mesh) -> Tensor:
    """Detect boundary vertices of a mesh.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.

    Returns
    -------
    Tensor
        Indices of boundary vertices, shape (num_boundary_vertices,).

    Examples
    --------
    >>> from torchscience.geometry.mesh import rectangle_mesh, mesh_boundary_vertices
    >>> mesh = rectangle_mesh(2, 2, bounds=[[0, 1], [0, 1]])
    >>> boundary_verts = mesh_boundary_vertices(mesh)
    >>> boundary_verts.shape
    torch.Size([8])
    """
    boundary_facets = mesh_boundary_facets(mesh)
    boundary_verts = torch.unique(boundary_facets.flatten())
    return boundary_verts


def _get_facet_local_indices(element_type: str) -> list[list[int]]:
    """Get local vertex indices for each facet of an element type."""
    if element_type == "line":
        return [[0], [1]]
    elif element_type == "triangle":
        return [[0, 1], [1, 2], [2, 0]]
    elif element_type == "quad":
        return [[0, 1], [1, 2], [2, 3], [3, 0]]
    elif element_type == "tetrahedron":
        return [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    elif element_type == "hexahedron":
        return [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5],  # right
        ]
    else:
        raise ValueError(f"Unknown element type: {element_type}")

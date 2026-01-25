"""Mesh refinement operators."""

from __future__ import annotations

import torch
from torch import Tensor

from ._mesh import Mesh


def refine_mesh(
    mesh: Mesh,
    level: int = 1,
) -> Mesh:
    """Uniformly refine a mesh by subdividing elements.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to refine.
    level : int, optional
        Number of refinement levels. Default 1.
        Each level quadruples the number of elements (for 2D).

    Returns
    -------
    Mesh
        Refined mesh with more elements.

    Notes
    -----
    For triangles: Each triangle is split into 4 triangles by adding
    midpoint vertices on each edge.

    For quads: Each quad is split into 4 quads by adding edge midpoints
    and a center point.

    Examples
    --------
    >>> from torchscience.geometry.mesh import rectangle_mesh, refine_mesh
    >>> mesh = rectangle_mesh(2, 2, bounds=[[0.0, 1.0], [0.0, 1.0]])
    >>> refined = refine_mesh(mesh, level=1)  # 32 triangles
    >>> refined.num_elements
    32

    """
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")

    if level == 0:
        return mesh

    # Apply refinement iteratively
    result = mesh
    for _ in range(level):
        result = _refine_once(result)

    return result


def _refine_once(mesh: Mesh) -> Mesh:
    """Apply one level of refinement to the mesh."""
    element_type = mesh.element_type

    if element_type == "triangle":
        return _refine_triangles(mesh)
    elif element_type == "quad":
        return _refine_quads(mesh)
    else:
        raise ValueError(
            f"Refinement not supported for element type: {element_type}"
        )


def _refine_triangles(mesh: Mesh) -> Mesh:
    """Refine a triangle mesh by subdividing each triangle into 4.

    Refinement pattern:
        Original:          Refined:
            2                  2
           / \\                /|\\
          /   \\       =>     / | \\
         /     \\            /  |  \\
        0-------1          0---3---1

        Each edge gets a midpoint (3, 4, 5 for edges 01, 12, 20).
        4 new triangles: (0,3,5), (3,1,4), (5,4,2), (3,4,5)
    """
    vertices = mesh.vertices
    elements = mesh.elements
    device = vertices.device
    dtype = vertices.dtype

    num_vertices = vertices.shape[0]
    num_elements = elements.shape[0]

    # Step 1: Collect all unique edges and create edge-to-midpoint mapping
    edges, edge_to_index = _collect_edges_triangles(elements)
    num_edges = edges.shape[0]

    # Step 2: Create new vertices at edge midpoints
    midpoint_vertices = _compute_midpoints(vertices, edges)

    # Step 3: Combine original and new vertices
    new_vertices = torch.cat([vertices, midpoint_vertices], dim=0)

    # Step 4: Create new element connectivity
    # For each triangle with vertices (v0, v1, v2) and midpoints (m01, m12, m20),
    # we create 4 triangles:
    # - (v0, m01, m20)  - corner triangle at v0
    # - (m01, v1, m12)  - corner triangle at v1
    # - (m20, m12, v2)  - corner triangle at v2
    # - (m01, m12, m20) - center triangle

    new_elements = torch.zeros(
        (num_elements * 4, 3), dtype=torch.int64, device=device
    )

    for i in range(num_elements):
        v0, v1, v2 = (
            elements[i, 0].item(),
            elements[i, 1].item(),
            elements[i, 2].item(),
        )

        # Get midpoint vertex indices (offset by num_vertices)
        m01 = num_vertices + edge_to_index[_edge_key(v0, v1)]
        m12 = num_vertices + edge_to_index[_edge_key(v1, v2)]
        m20 = num_vertices + edge_to_index[_edge_key(v2, v0)]

        # 4 new triangles
        new_elements[4 * i + 0] = torch.tensor(
            [v0, m01, m20], dtype=torch.int64, device=device
        )
        new_elements[4 * i + 1] = torch.tensor(
            [m01, v1, m12], dtype=torch.int64, device=device
        )
        new_elements[4 * i + 2] = torch.tensor(
            [m20, m12, v2], dtype=torch.int64, device=device
        )
        new_elements[4 * i + 3] = torch.tensor(
            [m01, m12, m20], dtype=torch.int64, device=device
        )

    return Mesh(
        vertices=new_vertices,
        elements=new_elements,
        element_type="triangle",
        boundary_facets=None,  # Will be recomputed by mesh_boundary_facets if needed
        facet_to_element=None,
        batch_size=[],
    )


def _refine_quads(mesh: Mesh) -> Mesh:
    """Refine a quad mesh by subdividing each quad into 4.

    Refinement pattern:
        Original:              Refined:
        3-------2              3---m2---2
        |       |              |   |    |
        |       |      =>      m3--c---m1
        |       |              |   |    |
        0-------1              0---m0---1

        Each edge gets a midpoint (m0, m1, m2, m3).
        Each quad gets a center point (c).
        4 new quads: (0,m0,c,m3), (m0,1,m1,c), (c,m1,2,m2), (m3,c,m2,3)
    """
    vertices = mesh.vertices
    elements = mesh.elements
    device = vertices.device
    dtype = vertices.dtype

    num_vertices = vertices.shape[0]
    num_elements = elements.shape[0]

    # Step 1: Collect all unique edges and create edge-to-midpoint mapping
    edges, edge_to_index = _collect_edges_quads(elements)
    num_edges = edges.shape[0]

    # Step 2: Create new vertices at edge midpoints
    midpoint_vertices = _compute_midpoints(vertices, edges)

    # Step 3: Compute quad centers
    center_vertices = torch.zeros(
        (num_elements, vertices.shape[1]), dtype=dtype, device=device
    )
    for i in range(num_elements):
        quad_verts = vertices[elements[i]]  # (4, dim)
        center_vertices[i] = quad_verts.mean(dim=0)

    # Step 4: Combine original, midpoint, and center vertices
    # Layout: [original vertices | edge midpoints | quad centers]
    new_vertices = torch.cat(
        [vertices, midpoint_vertices, center_vertices], dim=0
    )

    # Index offsets
    midpoint_offset = num_vertices
    center_offset = num_vertices + num_edges

    # Step 5: Create new element connectivity
    new_elements = torch.zeros(
        (num_elements * 4, 4), dtype=torch.int64, device=device
    )

    for i in range(num_elements):
        v0, v1, v2, v3 = (
            elements[i, 0].item(),
            elements[i, 1].item(),
            elements[i, 2].item(),
            elements[i, 3].item(),
        )

        # Get midpoint vertex indices
        m0 = midpoint_offset + edge_to_index[_edge_key(v0, v1)]  # bottom edge
        m1 = midpoint_offset + edge_to_index[_edge_key(v1, v2)]  # right edge
        m2 = midpoint_offset + edge_to_index[_edge_key(v2, v3)]  # top edge
        m3 = midpoint_offset + edge_to_index[_edge_key(v3, v0)]  # left edge

        # Center vertex index
        c = center_offset + i

        # 4 new quads (counter-clockwise ordering preserved)
        new_elements[4 * i + 0] = torch.tensor(
            [v0, m0, c, m3], dtype=torch.int64, device=device
        )
        new_elements[4 * i + 1] = torch.tensor(
            [m0, v1, m1, c], dtype=torch.int64, device=device
        )
        new_elements[4 * i + 2] = torch.tensor(
            [c, m1, v2, m2], dtype=torch.int64, device=device
        )
        new_elements[4 * i + 3] = torch.tensor(
            [m3, c, m2, v3], dtype=torch.int64, device=device
        )

    return Mesh(
        vertices=new_vertices,
        elements=new_elements,
        element_type="quad",
        boundary_facets=None,
        facet_to_element=None,
        batch_size=[],
    )


def _edge_key(v0: int, v1: int) -> tuple[int, int]:
    """Create a canonical edge key (smaller index first)."""
    return (min(v0, v1), max(v0, v1))


def _collect_edges_triangles(elements: Tensor) -> tuple[Tensor, dict]:
    """Collect all unique edges from triangle elements.

    Returns
    -------
    edges : Tensor
        Unique edges, shape (num_edges, 2).
    edge_to_index : dict
        Mapping from edge key (v0, v1) to edge index.
    """
    device = elements.device
    edge_set: dict[tuple[int, int], int] = {}
    edges_list: list[tuple[int, int]] = []

    for i in range(elements.shape[0]):
        v0, v1, v2 = (
            elements[i, 0].item(),
            elements[i, 1].item(),
            elements[i, 2].item(),
        )

        # Three edges per triangle
        for edge in [(v0, v1), (v1, v2), (v2, v0)]:
            key = _edge_key(*edge)
            if key not in edge_set:
                edge_set[key] = len(edges_list)
                edges_list.append(key)

    edges = torch.tensor(edges_list, dtype=torch.int64, device=device)
    return edges, edge_set


def _collect_edges_quads(elements: Tensor) -> tuple[Tensor, dict]:
    """Collect all unique edges from quad elements.

    Returns
    -------
    edges : Tensor
        Unique edges, shape (num_edges, 2).
    edge_to_index : dict
        Mapping from edge key (v0, v1) to edge index.
    """
    device = elements.device
    edge_set: dict[tuple[int, int], int] = {}
    edges_list: list[tuple[int, int]] = []

    for i in range(elements.shape[0]):
        v0, v1, v2, v3 = (
            elements[i, 0].item(),
            elements[i, 1].item(),
            elements[i, 2].item(),
            elements[i, 3].item(),
        )

        # Four edges per quad
        for edge in [(v0, v1), (v1, v2), (v2, v3), (v3, v0)]:
            key = _edge_key(*edge)
            if key not in edge_set:
                edge_set[key] = len(edges_list)
                edges_list.append(key)

    edges = torch.tensor(edges_list, dtype=torch.int64, device=device)
    return edges, edge_set


def _compute_midpoints(vertices: Tensor, edges: Tensor) -> Tensor:
    """Compute midpoint vertices for all edges.

    Parameters
    ----------
    vertices : Tensor
        Vertex coordinates, shape (num_vertices, dim).
    edges : Tensor
        Edge vertex indices, shape (num_edges, 2).

    Returns
    -------
    Tensor
        Midpoint coordinates, shape (num_edges, dim).
    """
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    return 0.5 * (v0 + v1)

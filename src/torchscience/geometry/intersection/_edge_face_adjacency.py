"""Edge-face adjacency construction for triangle meshes."""

from __future__ import annotations

import torch
from torch import Tensor

from ._edge_sampling_result import EdgeFaceAdjacency


def edge_face_adjacency(faces: Tensor) -> EdgeFaceAdjacency:
    r"""Construct edge-face adjacency from a triangle mesh face array.

    Extracts all unique edges from a triangle mesh and maps each edge to its
    one or two adjacent faces. This is a fundamental mesh topology query used
    in silhouette detection, boundary extraction, and mesh validation.

    Mathematical Definition
    -----------------------
    Given a triangle mesh with face array :math:`F \in \mathbb{Z}^{N \times 3}`,
    each face :math:`f_i = (v_0, v_1, v_2)` contributes three half-edges:

    .. math::
        (v_0, v_1), \quad (v_1, v_2), \quad (v_2, v_0)

    Each half-edge is canonicalized so that the smaller vertex index comes
    first: :math:`(\min(v_a, v_b), \max(v_a, v_b))`. Unique canonical edges
    are identified and each is associated with one face (boundary edge) or
    two faces (interior edge).

    A mesh is *manifold* if every edge is shared by at most two faces. This
    function raises :class:`ValueError` for non-manifold edges.

    Parameters
    ----------
    faces : Tensor, shape (N, 3)
        Triangle face indices with dtype ``torch.long``. Each row contains
        three vertex indices defining a triangle.

    Returns
    -------
    EdgeFaceAdjacency
        Tensorclass containing:

        - **edges** -- ``(num_edges, 2)`` canonical vertex pairs with ``i < j``
        - **face_0** -- ``(num_edges,)`` first adjacent face index
        - **face_1** -- ``(num_edges,)`` second adjacent face index (``-1`` if boundary)
        - **is_boundary** -- ``(num_edges,)`` boolean mask for boundary edges

    Raises
    ------
    ValueError
        If ``faces`` is not a 2D tensor with 3 columns, does not have dtype
        ``torch.long``, or contains non-manifold edges (shared by more than
        two faces).

    Examples
    --------
    Single triangle has three boundary edges:

    >>> faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    >>> adj = edge_face_adjacency(faces)
    >>> adj.edges.shape
    torch.Size([3, 2])
    >>> adj.is_boundary.all()
    tensor(True)

    Two triangles sharing an edge have one interior edge:

    >>> faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
    >>> adj = edge_face_adjacency(faces)
    >>> adj.edges.shape
    torch.Size([5, 2])
    >>> adj.is_boundary.sum()
    tensor(4)

    See Also
    --------
    EdgeFaceAdjacency : Tensorclass holding the adjacency result.
    """
    # --- Input validation ---
    if faces.ndim != 2:
        raise ValueError(
            f"faces must be a 2D tensor with shape (N, 3), got ndim={faces.ndim}"
        )

    if faces.shape[1] != 3:
        raise ValueError(
            f"faces must have 3 columns (triangles), got shape {faces.shape}"
        )

    if faces.dtype != torch.long:
        raise ValueError(
            f"faces must have dtype torch.long, got {faces.dtype}"
        )

    num_faces = faces.shape[0]
    device = faces.device

    # --- Step 1: Extract all 3 half-edges per face ---
    # (v0,v1), (v1,v2), (v2,v0) for each face
    v0 = faces[:, 0]  # (num_faces,)
    v1 = faces[:, 1]  # (num_faces,)
    v2 = faces[:, 2]  # (num_faces,)

    # Stack into (num_faces*3, 2) half-edge array
    half_edges = torch.stack(
        [
            torch.stack([v0, v1], dim=1),
            torch.stack([v1, v2], dim=1),
            torch.stack([v2, v0], dim=1),
        ],
        dim=0,
    )  # (3, num_faces, 2)
    half_edges = half_edges.reshape(-1, 2)  # (num_faces*3, 2)

    # Tag each half-edge with its source face index
    face_indices = torch.arange(num_faces, device=device).repeat(
        3
    )  # (num_faces*3,)

    # --- Step 2: Canonical ordering (min, max) ---
    edge_min = torch.min(half_edges[:, 0], half_edges[:, 1])
    edge_max = torch.max(half_edges[:, 0], half_edges[:, 1])

    # --- Step 3: Encode edges as integer keys ---
    max_vertex = faces.max().item() + 1
    keys = edge_min * max_vertex + edge_max  # (num_faces*3,) int64

    # --- Step 4: Sort by key and find unique edges ---
    sorted_order = torch.argsort(keys)
    sorted_keys = keys[sorted_order]
    sorted_face_indices = face_indices[sorted_order]
    sorted_edge_min = edge_min[sorted_order]
    sorted_edge_max = edge_max[sorted_order]

    unique_keys, counts = torch.unique(sorted_keys, return_counts=True)

    num_edges = unique_keys.shape[0]

    # --- Step 5: Validate manifoldness ---
    max_count = counts.max().item()
    if max_count > 2:
        raise ValueError(
            f"Non-manifold mesh detected: found edge(s) shared by {max_count} faces "
            f"(maximum allowed is 2)"
        )

    # --- Step 6: Build adjacency arrays ---
    # For each unique edge, find first and (optionally) second face.
    # Since keys are sorted, the first occurrence of each unique key is at
    # the position given by a cumulative count offset.

    # Compute the start index of each unique edge group in the sorted array
    # Using the inverse indices and first-occurrence logic:
    # We can find the first occurrence by checking where the sorted key changes.
    first_occurrence = torch.zeros(num_edges, dtype=torch.long, device=device)
    # The first element of each group: use cumsum of counts shifted by 1
    first_occurrence[0] = 0
    if num_edges > 1:
        first_occurrence[1:] = counts[:-1].cumsum(dim=0)

    # Extract canonical edge vertex pairs from the first occurrence
    edges_out = torch.stack(
        [sorted_edge_min[first_occurrence], sorted_edge_max[first_occurrence]],
        dim=1,
    )  # (num_edges, 2)

    # face_0 is the face from the first occurrence
    face_0 = sorted_face_indices[first_occurrence]  # (num_edges,)

    # face_1 is the face from the second occurrence (if count == 2), else -1
    face_1 = torch.full((num_edges,), -1, dtype=torch.long, device=device)
    has_second = counts == 2
    second_occurrence = first_occurrence + 1
    face_1[has_second] = sorted_face_indices[second_occurrence[has_second]]

    # --- Step 7: Boundary detection ---
    is_boundary = counts == 1  # (num_edges,) bool

    return EdgeFaceAdjacency(
        edges=edges_out,
        face_0=face_0,
        face_1=face_1,
        is_boundary=is_boundary,
        batch_size=[num_edges],
    )

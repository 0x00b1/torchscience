"""Silhouette edge detection for triangle meshes."""

from __future__ import annotations

import torch
from torch import Tensor

from ._edge_face_adjacency import edge_face_adjacency
from ._edge_sampling_result import EdgeFaceAdjacency, SilhouetteEdges


def silhouette_edges(
    vertices: Tensor,
    faces: Tensor,
    view_direction: Tensor,
    adjacency: EdgeFaceAdjacency | None = None,
) -> SilhouetteEdges:
    r"""Detect silhouette edges of a triangle mesh from a given view direction.

    Silhouette edges are mesh edges where one adjacent face is front-facing
    and the other is back-facing with respect to the view direction, or
    boundary edges whose single adjacent face is front-facing.

    This is a discrete (non-differentiable) selection operation. The
    downstream :func:`edge_sample` function provides differentiability
    for rendering applications.

    Mathematical Definition
    -----------------------
    Given face normals :math:`\mathbf{n}_f = (\mathbf{v}_1 - \mathbf{v}_0)
    \times (\mathbf{v}_2 - \mathbf{v}_0)` and view direction
    :math:`\mathbf{d}`, a face is classified as front-facing when:

    .. math::
        \mathbf{n}_f \cdot \mathbf{d} > 0

    An interior edge (shared by two faces) is a silhouette edge when one
    face is front-facing and the other is back-facing. A boundary edge
    (adjacent to only one face) is a silhouette edge when its single
    adjacent face is front-facing.

    Parameters
    ----------
    vertices : Tensor, shape (num_vertices, 3)
        Vertex positions of the triangle mesh.
    faces : Tensor, shape (num_faces, 3)
        Triangle face indices with dtype ``torch.long``.
    view_direction : Tensor, shape (3,) or (num_faces, 3)
        View direction vector(s). If shape ``(3,)``, the same direction is
        used for all faces. If shape ``(num_faces, 3)``, a per-face
        direction is used.
    adjacency : EdgeFaceAdjacency or None, optional
        Precomputed edge-face adjacency. If ``None``, it is computed from
        ``faces`` using :func:`edge_face_adjacency`.

    Returns
    -------
    SilhouetteEdges
        Tensorclass containing:

        - **edge_indices** -- ``(num_sil,)`` indices into adjacency edges
        - **edges** -- ``(num_sil, 2)`` vertex index pairs
        - **front_face** -- ``(num_sil,)`` front-facing face index
        - **back_face** -- ``(num_sil,)`` back-facing face index (``-1``
          for boundary edges)

    Raises
    ------
    ValueError
        If ``vertices`` is not a 2D tensor with 3 columns, ``faces`` is
        not a 2D tensor with 3 columns and dtype ``torch.long``, or
        ``view_direction`` has an invalid shape.

    Examples
    --------
    Single front-facing triangle has 3 boundary silhouette edges:

    >>> vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    >>> faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    >>> view = torch.tensor([0., 0., 1.])
    >>> result = silhouette_edges(vertices, faces, view)
    >>> result.edge_indices.shape[0]
    3

    See Also
    --------
    edge_face_adjacency : Construct edge-face adjacency from faces.
    """
    # --- Input validation ---
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"vertices must have shape (num_vertices, 3), got {vertices.shape}"
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"faces must have shape (num_faces, 3), got {faces.shape}"
        )

    if faces.dtype != torch.long:
        raise ValueError(
            f"faces must have dtype torch.long, got {faces.dtype}"
        )

    num_faces = faces.shape[0]

    if view_direction.ndim == 1:
        if view_direction.shape[0] != 3:
            raise ValueError(
                f"view_direction with ndim=1 must have shape (3,), "
                f"got {view_direction.shape}"
            )
    elif view_direction.ndim == 2:
        if view_direction.shape != (num_faces, 3):
            raise ValueError(
                f"view_direction with ndim=2 must have shape "
                f"({num_faces}, 3), got {view_direction.shape}"
            )
    else:
        raise ValueError(
            f"view_direction must be 1D or 2D, got ndim={view_direction.ndim}"
        )

    # --- Step 1: Compute adjacency if not provided ---
    if adjacency is None:
        adjacency = edge_face_adjacency(faces)

    # --- Step 2: Compute unnormalized face normals ---
    v0 = vertices[faces[:, 0]]  # (num_faces, 3)
    v1 = vertices[faces[:, 1]]  # (num_faces, 3)
    v2 = vertices[faces[:, 2]]  # (num_faces, 3)

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (num_faces, 3)

    # --- Step 3: Classify faces as front-facing ---
    if view_direction.ndim == 1:
        # Broadcast: dot product of each face normal with the single direction
        dots = (face_normals * view_direction.unsqueeze(0)).sum(dim=1)
    else:
        # Per-face dot product
        dots = (face_normals * view_direction).sum(dim=1)

    is_front = dots > 0  # (num_faces,) bool

    # --- Step 4: Classify edges ---
    front_0 = is_front[adjacency.face_0]  # (num_edges,)

    # For boundary edges, face_1 is -1. We need to handle indexing safely.
    # Clamp face_1 to 0 for safe indexing, then mask the result.
    safe_face_1 = adjacency.face_1.clamp(min=0)
    front_1 = is_front[safe_face_1]  # (num_edges,)
    # For boundary edges, front_1 is meaningless; set to False.
    front_1 = front_1 & ~adjacency.is_boundary

    # Interior silhouette: not boundary and front_0 != front_1
    interior_silhouette = ~adjacency.is_boundary & (front_0 != front_1)

    # Boundary silhouette: boundary and front-facing
    boundary_silhouette = adjacency.is_boundary & front_0

    # Combined silhouette mask
    is_silhouette = interior_silhouette | boundary_silhouette

    # --- Step 5: Gather silhouette edges ---
    sil_indices = is_silhouette.nonzero(as_tuple=True)[0]  # (num_sil,)
    num_sil = sil_indices.shape[0]

    sil_edges = adjacency.edges[sil_indices]  # (num_sil, 2)

    # Determine front_face and back_face for each silhouette edge
    sil_face_0 = adjacency.face_0[sil_indices]  # (num_sil,)
    sil_face_1 = adjacency.face_1[sil_indices]  # (num_sil,)
    sil_front_0 = front_0[sil_indices]  # (num_sil,) bool
    sil_is_boundary = adjacency.is_boundary[sil_indices]  # (num_sil,)

    # For interior silhouette edges: front_face is whichever face is
    # front-facing, back_face is the other.
    # For boundary silhouette edges: front_face is face_0, back_face is -1.
    front_face = torch.where(sil_front_0, sil_face_0, sil_face_1)
    back_face = torch.where(sil_front_0, sil_face_1, sil_face_0)

    # For boundary edges, force back_face = -1
    back_face = torch.where(
        sil_is_boundary,
        torch.tensor(-1, dtype=torch.long, device=faces.device),
        back_face,
    )

    return SilhouetteEdges(
        edge_indices=sil_indices,
        edges=sil_edges,
        front_face=front_face,
        back_face=back_face,
        batch_size=[num_sil],
    )

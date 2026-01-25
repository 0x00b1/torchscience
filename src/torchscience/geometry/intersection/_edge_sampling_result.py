"""Edge sampling result tensorclasses for differentiable rendering."""

from __future__ import annotations

from tensordict import tensorclass
from torch import Tensor


@tensorclass
class EdgeFaceAdjacency:
    """Edge-face adjacency structure for a triangle mesh.

    Maps each unique edge to its one or two adjacent faces. Boundary edges
    have exactly one adjacent face.

    Attributes
    ----------
    edges : Tensor, shape (num_edges, 2)
        Canonical vertex index pairs with i < j for each edge.
    face_0 : Tensor, shape (num_edges,)
        Index of the first adjacent face for each edge.
    face_1 : Tensor, shape (num_edges,)
        Index of the second adjacent face for each edge. Set to -1 for
        boundary edges that have only one adjacent face.
    is_boundary : Tensor, shape (num_edges,)
        Boolean mask indicating boundary edges (edges with only one
        adjacent face).
    """

    edges: Tensor
    face_0: Tensor
    face_1: Tensor
    is_boundary: Tensor


@tensorclass
class SilhouetteEdges:
    """Silhouette edges detected from a given viewpoint.

    Silhouette edges are mesh edges where one adjacent face is front-facing
    and the other is back-facing with respect to the camera, or boundary
    edges whose single adjacent face is front-facing.

    Attributes
    ----------
    edge_indices : Tensor, shape (num_sil,)
        Indices into the EdgeFaceAdjacency edges array identifying which
        mesh edges are silhouette edges.
    edges : Tensor, shape (num_sil, 2)
        Vertex index pairs for each silhouette edge.
    front_face : Tensor, shape (num_sil,)
        Index of the front-facing adjacent face for each silhouette edge.
    back_face : Tensor, shape (num_sil,)
        Index of the back-facing adjacent face for each silhouette edge.
        Set to -1 for boundary silhouette edges.
    """

    edge_indices: Tensor
    edges: Tensor
    front_face: Tensor
    back_face: Tensor


@tensorclass
class EdgeSamples:
    """Samples along silhouette edges for differentiable rendering.

    Each sample is a point on a silhouette edge, parameterized by a scalar
    t in [0, 1] that interpolates between the two edge vertices.

    Attributes
    ----------
    positions : Tensor, shape (num_samples, 3)
        World-space 3D positions of the sample points.
    edge_indices : Tensor, shape (num_samples,)
        Index of the parent silhouette edge for each sample.
    parametric_t : Tensor, shape (num_samples,)
        Parametric coordinate t in [0, 1] along the parent edge.
    edge_tangent : Tensor, shape (num_samples, 3)
        Unit tangent vector along the parent edge at each sample.
    edge_length : Tensor, shape (num_samples,)
        Length of the parent edge for each sample.
    """

    positions: Tensor
    edge_indices: Tensor
    parametric_t: Tensor
    edge_tangent: Tensor
    edge_length: Tensor

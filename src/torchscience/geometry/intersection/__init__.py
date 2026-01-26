"""Ray-geometry intersection primitives."""

from ._edge_face_adjacency import edge_face_adjacency
from ._edge_gradient import edge_gradient_contribution
from ._edge_sample import edge_sample
from ._edge_sampling_result import (
    EdgeFaceAdjacency,
    EdgeSamples,
    SilhouetteEdges,
)
from ._ray_aabb import ray_aabb
from ._ray_hit import IntersectionResult
from ._ray_plane import ray_plane
from ._ray_sphere import ray_sphere
from ._ray_sphere_intersection import RaySphereHit, ray_sphere_intersection
from ._ray_triangle import ray_triangle
from ._silhouette_edges import silhouette_edges

__all__ = [
    "EdgeFaceAdjacency",
    "EdgeSamples",
    "IntersectionResult",
    "RaySphereHit",
    "SilhouetteEdges",
    "edge_face_adjacency",
    "edge_gradient_contribution",
    "edge_sample",
    "ray_aabb",
    "ray_plane",
    "ray_sphere",
    "ray_sphere_intersection",
    "ray_triangle",
    "silhouette_edges",
]

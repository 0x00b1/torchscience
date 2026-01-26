"""
Intersection
============
"""

from ._edge_face_adjacency import edge_face_adjacency
from ._edge_gradient import edge_gradient_contribution
from ._edge_sample import edge_sample
from ._edge_sampling_result import (
    EdgeFaceAdjacency,
    EdgeSamples,
    SilhouetteEdges,
)
from ._ray_sphere_intersection import RaySphereHit, ray_sphere_intersection
from ._silhouette_edges import silhouette_edges

__all__ = [
    "EdgeFaceAdjacency",
    "EdgeSamples",
    "RaySphereHit",
    "SilhouetteEdges",
    "edge_face_adjacency",
    "edge_gradient_contribution",
    "edge_sample",
    "ray_sphere_intersection",
    "silhouette_edges",
]

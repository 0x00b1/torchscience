"""Ray-geometry intersection primitives."""

from ._ray_hit import IntersectionResult
from ._ray_plane import ray_plane
from ._ray_sphere import ray_sphere
from ._ray_sphere_intersection import RaySphereHit, ray_sphere_intersection
from ._ray_triangle import ray_triangle

__all__ = [
    "IntersectionResult",
    "RaySphereHit",
    "ray_plane",
    "ray_sphere",
    "ray_sphere_intersection",
    "ray_triangle",
]

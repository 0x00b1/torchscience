"""Ray-geometry intersection primitives."""

from ._ray_hit import IntersectionResult
from ._ray_sphere_intersection import RaySphereHit, ray_sphere_intersection

__all__ = [
    "IntersectionResult",
    "RaySphereHit",
    "ray_sphere_intersection",
]

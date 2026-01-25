"""Ray-geometry intersection primitives."""

from ._ray_hit import RayHit
from ._ray_sphere_intersection import RaySphereHit, ray_sphere_intersection

__all__ = [
    "RayHit",
    "RaySphereHit",
    "ray_sphere_intersection",
]

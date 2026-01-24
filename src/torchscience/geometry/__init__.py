"""Geometric operations and queries."""

from ._closest_point import ClosestPoint, closest_point
from ._convex_hull import ConvexHull, convex_hull
from ._exceptions import (
    DegenerateInputError,
    GeometryError,
    InsufficientPointsError,
)
from ._ray_intersect import RayHit, ray_intersect
from ._ray_occluded import ray_occluded
from .intersection import RaySphereHit, ray_sphere_intersection
from .mesh import Mesh, rectangle_mesh

__all__ = [
    "ClosestPoint",
    "ConvexHull",
    "DegenerateInputError",
    "GeometryError",
    "InsufficientPointsError",
    "Mesh",
    "RayHit",
    "RaySphereHit",
    "closest_point",
    "convex_hull",
    "ray_intersect",
    "ray_occluded",
    "ray_sphere_intersection",
    "rectangle_mesh",
]

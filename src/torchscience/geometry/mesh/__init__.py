"""Mesh data structures for finite element methods."""

from ._box_mesh import box_mesh
from ._mesh import Mesh
from ._rectangle_mesh import rectangle_mesh

__all__ = [
    "Mesh",
    "box_mesh",
    "rectangle_mesh",
]

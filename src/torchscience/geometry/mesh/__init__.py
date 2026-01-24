"""Mesh data structures for finite element methods."""

from ._mesh import Mesh
from ._rectangle_mesh import rectangle_mesh

__all__ = [
    "Mesh",
    "rectangle_mesh",
]

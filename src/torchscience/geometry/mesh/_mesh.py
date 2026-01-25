"""Mesh tensorclass for finite element computations."""

from __future__ import annotations

from typing import ClassVar

from tensordict import tensorclass
from torch import Tensor


@tensorclass
class Mesh:
    """Finite element mesh representation.

    As a tensorclass, Mesh supports:
    - Device movement: `mesh.to("cuda")` or `mesh.cuda()`
    - Dtype conversion: `mesh.to(torch.float64)`
    - Serialization: `torch.save(mesh, path)` / `torch.load(path)`

    Attributes
    ----------
    vertices : Tensor
        Vertex coordinates, shape (num_vertices, dim).
    elements : Tensor
        Element connectivity, shape (num_elements, nodes_per_element).
        Each row contains vertex indices forming an element.
    element_type : str
        Element type: "line", "triangle", "quad", "tetrahedron", "hexahedron".
    boundary_facets : Tensor | None
        Boundary facet connectivity, shape (num_boundary_facets, nodes_per_facet).
        For lines, facets are vertices (1 node).
        For triangles/quads, facets are edges (2 nodes).
        For tetrahedra, facets are triangles (3 nodes).
        For hexahedra, facets are quads (4 nodes).
    facet_to_element : Tensor | None
        Maps each boundary facet to its parent element, shape (num_boundary_facets,).

    Notes
    -----
    Element types and their properties:

    - "line": 1D, 2 nodes per element, facets are vertices (1 node)
    - "triangle": 2D, 3 nodes per element, facets are edges (2 nodes)
    - "quad": 2D, 4 nodes per element, facets are edges (2 nodes)
    - "tetrahedron": 3D, 4 nodes per element, facets are triangles (3 nodes)
    - "hexahedron": 3D, 8 nodes per element, facets are quads (4 nodes)
    """

    _NODES_PER_ELEMENT: ClassVar[dict] = {
        "line": 2,
        "triangle": 3,
        "quad": 4,
        "tetrahedron": 4,
        "hexahedron": 8,
    }

    _ELEMENT_DIM: ClassVar[dict] = {
        "line": 1,
        "triangle": 2,
        "quad": 2,
        "tetrahedron": 3,
        "hexahedron": 3,
    }

    vertices: Tensor
    elements: Tensor
    element_type: str
    boundary_facets: Tensor | None
    facet_to_element: Tensor | None

    @property
    def dim(self) -> int:
        """Spatial dimension of the mesh (2 or 3)."""
        return self.vertices.shape[-1]

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return self.vertices.shape[-2]

    @property
    def num_elements(self) -> int:
        """Number of elements in the mesh."""
        return self.elements.shape[-2]

    @property
    def nodes_per_element(self) -> int:
        """Number of nodes per element."""
        return self.elements.shape[-1]

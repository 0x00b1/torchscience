"""Rectangle mesh generator for 2D finite element meshes."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ._mesh import Mesh


def rectangle_mesh(
    nx: int,
    ny: int,
    bounds: Tensor | Sequence[Sequence[float]],
    element_type: str = "triangle",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Mesh:
    """Generate a structured mesh on a rectangular domain.

    Creates a mesh of triangular or quadrilateral elements on a 2D rectangular
    domain. The mesh is structured with nx elements in the x-direction and
    ny elements in the y-direction.

    Parameters
    ----------
    nx : int
        Number of elements in the x-direction. Must be >= 1.
    ny : int
        Number of elements in the y-direction. Must be >= 1.
    bounds : Tensor | Sequence[Sequence[float]]
        Domain bounds as [[x_min, x_max], [y_min, y_max]].
        Can be a tensor with shape (2, 2) for differentiable mesh generation.
    element_type : str, optional
        Element type: "triangle" or "quad". Default is "triangle".
    dtype : torch.dtype, optional
        Data type for vertex coordinates. Default is torch.float64.
    device : torch.device, optional
        Device for mesh tensors. Default is CPU.

    Returns
    -------
    Mesh
        A Mesh tensorclass instance with:
        - vertices: shape ((nx+1)*(ny+1), 2)
        - elements: shape (nx*ny*2, 3) for triangles or (nx*ny, 4) for quads
        - boundary_facets: boundary edges of the mesh
        - facet_to_element: mapping from boundary facets to parent elements

    Notes
    -----
    For triangle meshes, each rectangular cell is split into two triangles
    using a diagonal from the lower-left to upper-right corner.

    The mesh is differentiable with respect to the bounds tensor, which
    enables shape optimization applications.

    Examples
    --------
    >>> import torch
    >>> from torchscience.geometry.mesh import rectangle_mesh
    >>> mesh = rectangle_mesh(2, 2, [[0.0, 1.0], [0.0, 1.0]], "triangle")
    >>> mesh.num_vertices
    9
    >>> mesh.num_elements
    8

    """
    # Input validation
    if nx < 1:
        raise ValueError(f"nx must be >= 1, got {nx}")
    if ny < 1:
        raise ValueError(f"ny must be >= 1, got {ny}")
    if element_type not in ("triangle", "quad"):
        raise ValueError(
            f"element_type must be 'triangle' or 'quad', got '{element_type}'"
        )

    # Set defaults
    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Convert bounds to tensor if needed
    if not isinstance(bounds, Tensor):
        bounds = torch.tensor(bounds, dtype=dtype, device=device)
    else:
        # Ensure bounds is on the correct device and dtype
        bounds = bounds.to(dtype=dtype, device=device)

    # Extract bounds
    x_min, x_max = bounds[0, 0], bounds[0, 1]
    y_min, y_max = bounds[1, 0], bounds[1, 1]

    # Generate vertex grid
    # Using linspace for uniform spacing
    x_coords = torch.linspace(0.0, 1.0, nx + 1, dtype=dtype, device=device)
    y_coords = torch.linspace(0.0, 1.0, ny + 1, dtype=dtype, device=device)

    # Scale to actual bounds (preserves gradients)
    x_coords = x_min + x_coords * (x_max - x_min)
    y_coords = y_min + y_coords * (y_max - y_min)

    # Create vertex grid using meshgrid
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing="xy")
    vertices = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    # Generate element connectivity
    elements = _generate_elements(nx, ny, element_type, device)

    # Generate boundary facets
    boundary_facets, facet_to_element = _generate_boundary_facets(
        nx, ny, element_type, device
    )

    return Mesh(
        vertices=vertices,
        elements=elements,
        element_type=element_type,
        boundary_facets=boundary_facets,
        facet_to_element=facet_to_element,
        batch_size=[],
    )


def _generate_elements(
    nx: int,
    ny: int,
    element_type: str,
    device: torch.device,
) -> Tensor:
    """Generate element connectivity for a structured grid.

    Vertex numbering follows row-major order:
    - Vertex (i, j) has index j * (nx + 1) + i
    - where i is the x-index (0 to nx) and j is the y-index (0 to ny)
    """
    elements = []

    for j in range(ny):
        for i in range(nx):
            # Indices of the four corners of cell (i, j)
            # Bottom-left, bottom-right, top-right, top-left
            v00 = j * (nx + 1) + i  # (i, j)
            v10 = j * (nx + 1) + (i + 1)  # (i+1, j)
            v11 = (j + 1) * (nx + 1) + (i + 1)  # (i+1, j+1)
            v01 = (j + 1) * (nx + 1) + i  # (i, j+1)

            if element_type == "triangle":
                # Split quad into two triangles using diagonal from v00 to v11
                # Lower triangle: v00 -> v10 -> v11
                # Upper triangle: v00 -> v11 -> v01
                elements.append([v00, v10, v11])
                elements.append([v00, v11, v01])
            else:  # quad
                # Counter-clockwise ordering: v00 -> v10 -> v11 -> v01
                elements.append([v00, v10, v11, v01])

    return torch.tensor(elements, dtype=torch.int64, device=device)


def _generate_boundary_facets(
    nx: int,
    ny: int,
    element_type: str,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Generate boundary facets (edges) for a structured grid.

    Returns boundary edges in the following order:
    - Bottom edge (y=0): left to right
    - Right edge (x=1): bottom to top
    - Top edge (y=1): right to left
    - Left edge (x=0): top to bottom

    Also returns the parent element index for each boundary facet.
    """
    boundary_facets = []
    facet_to_element = []

    # Bottom edge (y=0)
    for i in range(nx):
        v0 = i  # (i, 0)
        v1 = i + 1  # (i+1, 0)
        boundary_facets.append([v0, v1])
        # Parent element is the first triangle/quad in cell (i, 0)
        if element_type == "triangle":
            facet_to_element.append(2 * (0 * nx + i))  # lower triangle
        else:
            facet_to_element.append(0 * nx + i)

    # Right edge (x=1)
    for j in range(ny):
        v0 = j * (nx + 1) + nx  # (nx, j)
        v1 = (j + 1) * (nx + 1) + nx  # (nx, j+1)
        boundary_facets.append([v0, v1])
        if element_type == "triangle":
            # Right edge belongs to the lower triangle
            facet_to_element.append(2 * (j * nx + (nx - 1)))  # lower triangle
        else:
            facet_to_element.append(j * nx + (nx - 1))

    # Top edge (y=1)
    for i in range(nx - 1, -1, -1):
        v0 = ny * (nx + 1) + (i + 1)  # (i+1, ny)
        v1 = ny * (nx + 1) + i  # (i, ny)
        boundary_facets.append([v0, v1])
        if element_type == "triangle":
            # Top edge belongs to the upper triangle
            facet_to_element.append(
                2 * ((ny - 1) * nx + i) + 1
            )  # upper triangle
        else:
            facet_to_element.append((ny - 1) * nx + i)

    # Left edge (x=0)
    for j in range(ny - 1, -1, -1):
        v0 = (j + 1) * (nx + 1)  # (0, j+1)
        v1 = j * (nx + 1)  # (0, j)
        boundary_facets.append([v0, v1])
        if element_type == "triangle":
            # Left edge belongs to the upper triangle
            facet_to_element.append(2 * (j * nx) + 1)  # upper triangle
        else:
            facet_to_element.append(j * nx)

    return (
        torch.tensor(boundary_facets, dtype=torch.int64, device=device),
        torch.tensor(facet_to_element, dtype=torch.int64, device=device),
    )

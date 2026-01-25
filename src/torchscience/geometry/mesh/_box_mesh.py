"""Box mesh generator for 3D finite element meshes."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ._mesh import Mesh


def box_mesh(
    nx: int,
    ny: int,
    nz: int,
    bounds: Tensor | Sequence[Sequence[float]],
    element_type: str = "tetrahedron",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Mesh:
    """Generate a structured mesh on a box (3D) domain.

    Creates a mesh of tetrahedral or hexahedral elements on a 3D box domain.
    The mesh is structured with nx elements in the x-direction, ny elements
    in the y-direction, and nz elements in the z-direction.

    Parameters
    ----------
    nx : int
        Number of elements in the x-direction. Must be >= 1.
    ny : int
        Number of elements in the y-direction. Must be >= 1.
    nz : int
        Number of elements in the z-direction. Must be >= 1.
    bounds : Tensor | Sequence[Sequence[float]]
        Domain bounds as [[x_min, x_max], [y_min, y_max], [z_min, z_max]].
        Can be a tensor with shape (3, 2) for differentiable mesh generation.
    element_type : str, optional
        Element type: "tetrahedron" or "hexahedron". Default is "tetrahedron".
    dtype : torch.dtype, optional
        Data type for vertex coordinates. Default is torch.float64.
    device : torch.device, optional
        Device for mesh tensors. Default is CPU.

    Returns
    -------
    Mesh
        A Mesh tensorclass instance with:
        - vertices: shape ((nx+1)*(ny+1)*(nz+1), 3)
        - elements: shape (nx*ny*nz*6, 4) for tetrahedra or (nx*ny*nz, 8) for hexahedra
        - boundary_facets: boundary faces of the mesh
        - facet_to_element: mapping from boundary facets to parent elements

    Notes
    -----
    For tetrahedral meshes, each cube is decomposed into 6 tetrahedra using
    a consistent decomposition that ensures conforming faces between adjacent
    cubes.

    The mesh is differentiable with respect to the bounds tensor, which
    enables shape optimization applications.

    Examples
    --------
    >>> import torch
    >>> from torchscience.geometry.mesh import box_mesh
    >>> mesh = box_mesh(2, 2, 2, [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], "tetrahedron")
    >>> mesh.num_vertices
    27
    >>> mesh.num_elements
    48

    """
    # Input validation
    if nx < 1:
        raise ValueError(f"nx must be >= 1, got {nx}")
    if ny < 1:
        raise ValueError(f"ny must be >= 1, got {ny}")
    if nz < 1:
        raise ValueError(f"nz must be >= 1, got {nz}")
    if element_type not in ("tetrahedron", "hexahedron"):
        raise ValueError(
            f"element_type must be 'tetrahedron' or 'hexahedron', got '{element_type}'"
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
    z_min, z_max = bounds[2, 0], bounds[2, 1]

    # Generate vertex grid
    # Using linspace for uniform spacing
    x_coords = torch.linspace(0.0, 1.0, nx + 1, dtype=dtype, device=device)
    y_coords = torch.linspace(0.0, 1.0, ny + 1, dtype=dtype, device=device)
    z_coords = torch.linspace(0.0, 1.0, nz + 1, dtype=dtype, device=device)

    # Scale to actual bounds (preserves gradients)
    x_coords = x_min + x_coords * (x_max - x_min)
    y_coords = y_min + y_coords * (y_max - y_min)
    z_coords = z_min + z_coords * (z_max - z_min)

    # Create vertex grid using meshgrid
    # Vertex index: k * (ny + 1) * (nx + 1) + j * (nx + 1) + i
    # where i is x-index, j is y-index, k is z-index
    xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    vertices = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)

    # Generate element connectivity
    elements = _generate_elements(nx, ny, nz, element_type, device)

    # Generate boundary facets
    boundary_facets, facet_to_element = _generate_boundary_facets(
        nx, ny, nz, element_type, device
    )

    return Mesh(
        vertices=vertices,
        elements=elements,
        element_type=element_type,
        boundary_facets=boundary_facets,
        facet_to_element=facet_to_element,
        batch_size=[],
    )


def _vertex_index(i: int, j: int, k: int, nx: int, ny: int) -> int:
    """Compute vertex index from grid coordinates.

    Vertex (i, j, k) has index k * (ny + 1) * (nx + 1) + j * (nx + 1) + i.
    """
    return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i


def _generate_elements(
    nx: int,
    ny: int,
    nz: int,
    element_type: str,
    device: torch.device,
) -> Tensor:
    """Generate element connectivity for a structured 3D grid.

    For tetrahedra, uses the 6-tet decomposition per cube.
    For hexahedra, uses VTK ordering (counter-clockwise bottom face, then top).
    """
    elements = []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Indices of the eight corners of cube (i, j, k)
                # Using convention: v_{abc} where a=x-offset, b=y-offset, c=z-offset
                v000 = _vertex_index(i, j, k, nx, ny)
                v100 = _vertex_index(i + 1, j, k, nx, ny)
                v010 = _vertex_index(i, j + 1, k, nx, ny)
                v110 = _vertex_index(i + 1, j + 1, k, nx, ny)
                v001 = _vertex_index(i, j, k + 1, nx, ny)
                v101 = _vertex_index(i + 1, j, k + 1, nx, ny)
                v011 = _vertex_index(i, j + 1, k + 1, nx, ny)
                v111 = _vertex_index(i + 1, j + 1, k + 1, nx, ny)

                if element_type == "tetrahedron":
                    # 6-tetrahedron decomposition of a cube
                    # This decomposition is consistent across adjacent cubes
                    # (all use the same diagonal v000-v111)
                    #
                    # The 6 tetrahedra are formed by connecting the main diagonal
                    # v000-v111 with each of the 6 faces of the cube.
                    #
                    # For positive orientation (positive volume), vertices are
                    # ordered so the fourth vertex sees the first three in
                    # counter-clockwise order.
                    #
                    # The tetrahedra correspond to the 6 faces of the cube:
                    # Face z=0 (bottom): v000, v100, v110, v010
                    # Face z=1 (top): v001, v011, v111, v101
                    # Face y=0 (front): v000, v001, v101, v100
                    # Face y=1 (back): v010, v110, v111, v011
                    # Face x=0 (left): v000, v010, v011, v001
                    # Face x=1 (right): v100, v101, v111, v110
                    elements.append([v000, v110, v100, v111])  # bottom-right
                    elements.append([v000, v010, v110, v111])  # bottom-left
                    elements.append([v000, v011, v010, v111])  # left-back
                    elements.append([v000, v001, v011, v111])  # left-front
                    elements.append([v000, v101, v001, v111])  # front-right
                    elements.append([v000, v100, v101, v111])  # right-bottom
                else:  # hexahedron
                    # VTK hexahedron ordering:
                    # Bottom face (z=k): v0, v1, v2, v3 counter-clockwise
                    # Top face (z=k+1): v4, v5, v6, v7 counter-clockwise
                    # v4 is above v0, v5 above v1, etc.
                    elements.append(
                        [v000, v100, v110, v010, v001, v101, v111, v011]
                    )

    return torch.tensor(elements, dtype=torch.int64, device=device)


def _generate_boundary_facets(
    nx: int,
    ny: int,
    nz: int,
    element_type: str,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Generate boundary facets for a structured 3D grid.

    Returns boundary faces and their parent element indices.

    For tetrahedra, boundary facets are triangles (3 nodes).
    For hexahedra, boundary facets are quads (4 nodes).
    """
    boundary_facets = []
    facet_to_element = []

    if element_type == "tetrahedron":
        # For tetrahedra, boundary faces are triangles
        # Each boundary quad is split into 2 triangles consistent with tet decomposition

        # z = 0 face (bottom)
        for j in range(ny):
            for i in range(nx):
                v00 = _vertex_index(i, j, 0, nx, ny)
                v10 = _vertex_index(i + 1, j, 0, nx, ny)
                v01 = _vertex_index(i, j + 1, 0, nx, ny)
                v11 = _vertex_index(i + 1, j + 1, 0, nx, ny)
                # Element index for cube (i, j, k=0)
                cube_idx = j * nx + i
                # Bottom face triangles from tetrahedra
                # Tet 0: v000, v100, v110, v111 -> bottom face: v000, v100, v110
                # Tet 1: v000, v110, v010, v111 -> bottom face: v000, v110, v010
                boundary_facets.append([v00, v10, v11])
                facet_to_element.append(cube_idx * 6 + 0)
                boundary_facets.append([v00, v11, v01])
                facet_to_element.append(cube_idx * 6 + 1)

        # z = nz face (top)
        for j in range(ny):
            for i in range(nx):
                v00 = _vertex_index(i, j, nz, nx, ny)
                v10 = _vertex_index(i + 1, j, nz, nx, ny)
                v01 = _vertex_index(i, j + 1, nz, nx, ny)
                v11 = _vertex_index(i + 1, j + 1, nz, nx, ny)
                # Element index for cube (i, j, k=nz-1)
                cube_idx = (nz - 1) * ny * nx + j * nx + i
                # Top face triangles (reversed for outward normal)
                # Tet 2: v000, v010, v011, v111 -> top face: v010, v011, v111
                # Tet 3: v000, v011, v001, v111 -> top face: v011, v001, v111
                boundary_facets.append([v00, v11, v10])
                facet_to_element.append(cube_idx * 6 + 4)
                boundary_facets.append([v00, v01, v11])
                facet_to_element.append(cube_idx * 6 + 3)

        # y = 0 face (front)
        for k in range(nz):
            for i in range(nx):
                v00 = _vertex_index(i, 0, k, nx, ny)
                v10 = _vertex_index(i + 1, 0, k, nx, ny)
                v01 = _vertex_index(i, 0, k + 1, nx, ny)
                v11 = _vertex_index(i + 1, 0, k + 1, nx, ny)
                cube_idx = k * ny * nx + i
                # y=0 face from tetrahedra 4, 5
                boundary_facets.append([v00, v11, v10])
                facet_to_element.append(cube_idx * 6 + 5)
                boundary_facets.append([v00, v01, v11])
                facet_to_element.append(cube_idx * 6 + 4)

        # y = ny face (back)
        for k in range(nz):
            for i in range(nx):
                v00 = _vertex_index(i, ny, k, nx, ny)
                v10 = _vertex_index(i + 1, ny, k, nx, ny)
                v01 = _vertex_index(i, ny, k + 1, nx, ny)
                v11 = _vertex_index(i + 1, ny, k + 1, nx, ny)
                cube_idx = k * ny * nx + (ny - 1) * nx + i
                # y=ny face from tetrahedra 1, 2
                boundary_facets.append([v00, v10, v11])
                facet_to_element.append(cube_idx * 6 + 1)
                boundary_facets.append([v00, v11, v01])
                facet_to_element.append(cube_idx * 6 + 2)

        # x = 0 face (left)
        for k in range(nz):
            for j in range(ny):
                v00 = _vertex_index(0, j, k, nx, ny)
                v10 = _vertex_index(0, j + 1, k, nx, ny)
                v01 = _vertex_index(0, j, k + 1, nx, ny)
                v11 = _vertex_index(0, j + 1, k + 1, nx, ny)
                cube_idx = k * ny * nx + j * nx
                # x=0 face from tetrahedra 2, 3
                boundary_facets.append([v00, v10, v11])
                facet_to_element.append(cube_idx * 6 + 2)
                boundary_facets.append([v00, v11, v01])
                facet_to_element.append(cube_idx * 6 + 3)

        # x = nx face (right)
        for k in range(nz):
            for j in range(ny):
                v00 = _vertex_index(nx, j, k, nx, ny)
                v10 = _vertex_index(nx, j + 1, k, nx, ny)
                v01 = _vertex_index(nx, j, k + 1, nx, ny)
                v11 = _vertex_index(nx, j + 1, k + 1, nx, ny)
                cube_idx = k * ny * nx + j * nx + (nx - 1)
                # x=nx face from tetrahedra 0, 5
                boundary_facets.append([v00, v11, v10])
                facet_to_element.append(cube_idx * 6 + 0)
                boundary_facets.append([v00, v01, v11])
                facet_to_element.append(cube_idx * 6 + 5)

    else:  # hexahedron
        # For hexahedra, boundary facets are quads

        # z = 0 face (bottom)
        for j in range(ny):
            for i in range(nx):
                v00 = _vertex_index(i, j, 0, nx, ny)
                v10 = _vertex_index(i + 1, j, 0, nx, ny)
                v01 = _vertex_index(i, j + 1, 0, nx, ny)
                v11 = _vertex_index(i + 1, j + 1, 0, nx, ny)
                cube_idx = j * nx + i
                # Bottom face, outward normal points -z
                boundary_facets.append([v00, v01, v11, v10])
                facet_to_element.append(cube_idx)

        # z = nz face (top)
        for j in range(ny):
            for i in range(nx):
                v00 = _vertex_index(i, j, nz, nx, ny)
                v10 = _vertex_index(i + 1, j, nz, nx, ny)
                v01 = _vertex_index(i, j + 1, nz, nx, ny)
                v11 = _vertex_index(i + 1, j + 1, nz, nx, ny)
                cube_idx = (nz - 1) * ny * nx + j * nx + i
                # Top face, outward normal points +z
                boundary_facets.append([v00, v10, v11, v01])
                facet_to_element.append(cube_idx)

        # y = 0 face (front)
        for k in range(nz):
            for i in range(nx):
                v00 = _vertex_index(i, 0, k, nx, ny)
                v10 = _vertex_index(i + 1, 0, k, nx, ny)
                v01 = _vertex_index(i, 0, k + 1, nx, ny)
                v11 = _vertex_index(i + 1, 0, k + 1, nx, ny)
                cube_idx = k * ny * nx + i
                # Front face, outward normal points -y
                boundary_facets.append([v00, v10, v11, v01])
                facet_to_element.append(cube_idx)

        # y = ny face (back)
        for k in range(nz):
            for i in range(nx):
                v00 = _vertex_index(i, ny, k, nx, ny)
                v10 = _vertex_index(i + 1, ny, k, nx, ny)
                v01 = _vertex_index(i, ny, k + 1, nx, ny)
                v11 = _vertex_index(i + 1, ny, k + 1, nx, ny)
                cube_idx = k * ny * nx + (ny - 1) * nx + i
                # Back face, outward normal points +y
                boundary_facets.append([v00, v01, v11, v10])
                facet_to_element.append(cube_idx)

        # x = 0 face (left)
        for k in range(nz):
            for j in range(ny):
                v00 = _vertex_index(0, j, k, nx, ny)
                v10 = _vertex_index(0, j + 1, k, nx, ny)
                v01 = _vertex_index(0, j, k + 1, nx, ny)
                v11 = _vertex_index(0, j + 1, k + 1, nx, ny)
                cube_idx = k * ny * nx + j * nx
                # Left face, outward normal points -x
                boundary_facets.append([v00, v01, v11, v10])
                facet_to_element.append(cube_idx)

        # x = nx face (right)
        for k in range(nz):
            for j in range(ny):
                v00 = _vertex_index(nx, j, k, nx, ny)
                v10 = _vertex_index(nx, j + 1, k, nx, ny)
                v01 = _vertex_index(nx, j, k + 1, nx, ny)
                v11 = _vertex_index(nx, j + 1, k + 1, nx, ny)
                cube_idx = k * ny * nx + j * nx + (nx - 1)
                # Right face, outward normal points +x
                boundary_facets.append([v00, v10, v11, v01])
                facet_to_element.append(cube_idx)

    return (
        torch.tensor(boundary_facets, dtype=torch.int64, device=device),
        torch.tensor(facet_to_element, dtype=torch.int64, device=device),
    )

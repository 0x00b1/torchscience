"""Delaunay triangulation using the Bowyer-Watson algorithm."""

from __future__ import annotations

import torch
from torch import Tensor

from ._mesh import Mesh


def delaunay_triangulation(points: Tensor) -> Mesh:
    """Create a Delaunay triangulation from a set of 2D points.

    Parameters
    ----------
    points : Tensor
        Point coordinates, shape (num_points, 2).

    Returns
    -------
    Mesh
        Triangle mesh with the Delaunay triangulation.

    Raises
    ------
    ValueError
        If points has fewer than 3 points, wrong dimensionality,
        or if all points are collinear.

    Notes
    -----
    Uses the Bowyer-Watson incremental algorithm.
    The resulting mesh satisfies the Delaunay criterion: no point lies
    inside any triangle's circumcircle.

    Examples
    --------
    >>> import torch
    >>> from torchscience.geometry.mesh import delaunay_triangulation
    >>> points = torch.rand(20, 2)
    >>> mesh = delaunay_triangulation(points)
    >>> mesh.element_type
    'triangle'
    """
    # Input validation
    if points.ndim != 2:
        raise ValueError(
            "points must be a 2D tensor with shape (num_points, 2)"
        )

    if points.shape[1] != 2:
        raise ValueError(
            "points must be 2D coordinates with shape (num_points, 2)"
        )

    num_points = points.shape[0]

    if num_points < 3:
        raise ValueError("Delaunay triangulation requires at least 3 points")

    # Convert to float64 for numerical stability during computation
    dtype = points.dtype
    device = points.device
    points_f64 = points.to(dtype=torch.float64)

    # Remove duplicate points
    unique_points, inverse_indices = _remove_duplicates(points_f64)
    num_unique = unique_points.shape[0]

    if num_unique < 3:
        raise ValueError(
            "Delaunay triangulation requires at least 3 non-duplicate points"
        )

    # Check for collinearity
    if _are_collinear(unique_points):
        raise ValueError("All points are collinear, cannot form triangulation")

    # Compute bounding box and create super-triangle
    min_coords = unique_points.min(dim=0).values
    max_coords = unique_points.max(dim=0).values
    delta = max_coords - min_coords
    delta_max = delta.max().item()

    # Create a super-triangle that contains all points
    # Make it large enough to avoid numerical issues
    margin = 10.0 * delta_max
    center = (min_coords + max_coords) / 2

    # Super-triangle vertices (equilateral-ish triangle containing all points)
    st0 = center + torch.tensor(
        [-margin * 2, -margin], dtype=torch.float64, device=device
    )
    st1 = center + torch.tensor(
        [margin * 2, -margin], dtype=torch.float64, device=device
    )
    st2 = center + torch.tensor(
        [0, margin * 2], dtype=torch.float64, device=device
    )

    # Initialize triangulation with super-triangle
    # Vertices: super-triangle vertices followed by actual points
    super_vertices = torch.stack([st0, st1, st2], dim=0)
    all_vertices = torch.cat([super_vertices, unique_points], dim=0)

    # Triangle storage: list of (v0, v1, v2) tuples (indices)
    # Super-triangle is triangle 0, 1, 2
    triangles = [(0, 1, 2)]

    # Insert points one at a time
    for i in range(num_unique):
        point_idx = i + 3  # Offset by super-triangle vertices
        point = all_vertices[point_idx]

        # Find bad triangles (those whose circumcircle contains the new point)
        bad_triangles = []
        for tri_idx, (v0, v1, v2) in enumerate(triangles):
            if _point_in_circumcircle(
                point,
                all_vertices[v0],
                all_vertices[v1],
                all_vertices[v2],
            ):
                bad_triangles.append(tri_idx)

        # Find the boundary of the cavity (edges that belong to only one bad triangle)
        edge_count: dict[tuple[int, int], int] = {}
        for tri_idx in bad_triangles:
            v0, v1, v2 = triangles[tri_idx]
            edges = [(v0, v1), (v1, v2), (v2, v0)]
            for edge in edges:
                # Normalize edge direction for consistent comparison
                edge_key = tuple(sorted(edge))
                edge_count[edge_key] = edge_count.get(edge_key, 0) + 1

        # Boundary edges appear only once
        boundary_edges = [
            edge for edge, count in edge_count.items() if count == 1
        ]

        # Remove bad triangles (iterate in reverse to avoid index shifting)
        for tri_idx in sorted(bad_triangles, reverse=True):
            triangles.pop(tri_idx)

        # Create new triangles by connecting the new point to boundary edges
        for e0, e1 in boundary_edges:
            # Orient triangle correctly (counter-clockwise)
            if _orient_2d(all_vertices[e0], all_vertices[e1], point) > 0:
                triangles.append((e0, e1, point_idx))
            else:
                triangles.append((e1, e0, point_idx))

    # Remove triangles that share vertices with the super-triangle
    super_triangle_vertices = {0, 1, 2}
    final_triangles = []
    for v0, v1, v2 in triangles:
        if (
            v0 not in super_triangle_vertices
            and v1 not in super_triangle_vertices
            and v2 not in super_triangle_vertices
        ):
            # Remap indices (subtract 3 to account for removed super-triangle vertices)
            final_triangles.append((v0 - 3, v1 - 3, v2 - 3))

    if len(final_triangles) == 0:
        raise ValueError("No valid triangles after removing super-triangle")

    # Create output tensors
    vertices = unique_points.to(dtype=dtype)
    elements = torch.tensor(final_triangles, dtype=torch.int64, device=device)

    return Mesh(
        vertices=vertices,
        elements=elements,
        element_type="triangle",
        boundary_facets=None,
        facet_to_element=None,
        batch_size=torch.Size([]),
    )


def _remove_duplicates(
    points: Tensor, tol: float = 1e-10
) -> tuple[Tensor, Tensor]:
    """Remove duplicate points within tolerance.

    Parameters
    ----------
    points : Tensor
        Input points, shape (n, 2).
    tol : float
        Tolerance for considering points equal.

    Returns
    -------
    unique_points : Tensor
        Points with duplicates removed, shape (m, 2) where m <= n.
    inverse_indices : Tensor
        Mapping from original indices to unique indices, shape (n,).
    """
    n = points.shape[0]
    device = points.device

    if n == 0:
        return points, torch.empty(0, dtype=torch.long, device=device)

    unique_indices = [0]
    inverse = torch.zeros(n, dtype=torch.long, device=device)

    for i in range(1, n):
        is_duplicate = False
        for j, unique_idx in enumerate(unique_indices):
            if torch.norm(points[i] - points[unique_idx]) < tol:
                inverse[i] = j
                is_duplicate = True
                break
        if not is_duplicate:
            inverse[i] = len(unique_indices)
            unique_indices.append(i)

    unique_points = points[unique_indices]
    return unique_points, inverse


def _are_collinear(points: Tensor, tol: float = 1e-10) -> bool:
    """Check if all points are collinear.

    Parameters
    ----------
    points : Tensor
        Input points, shape (n, 2).
    tol : float
        Tolerance for collinearity check.

    Returns
    -------
    bool
        True if all points are collinear.
    """
    n = points.shape[0]
    if n < 3:
        return True

    # Use first two points to define a reference direction
    p0 = points[0]
    p1 = points[1]
    ref_vec = p1 - p0
    ref_len = torch.norm(ref_vec)

    if ref_len < tol:
        # First two points are the same
        # Find another distinct point
        for i in range(2, n):
            if torch.norm(points[i] - p0) >= tol:
                ref_vec = points[i] - p0
                ref_len = torch.norm(ref_vec)
                break
        else:
            # All points are essentially the same
            return True

    ref_vec = ref_vec / ref_len

    # Check if all other points are collinear with p0 and the reference direction
    for i in range(n):
        vec = points[i] - p0
        # Cross product in 2D: v1.x * v2.y - v1.y * v2.x
        cross = ref_vec[0] * vec[1] - ref_vec[1] * vec[0]
        if abs(cross) > tol:
            return False

    return True


def _point_in_circumcircle(p: Tensor, a: Tensor, b: Tensor, c: Tensor) -> bool:
    """Check if point p is inside the circumcircle of triangle (a, b, c).

    Uses the incircle predicate: computes the sign of the determinant of:
    | ax-px  ay-py  (ax-px)^2 + (ay-py)^2 |
    | bx-px  by-py  (bx-px)^2 + (by-py)^2 |
    | cx-px  cy-py  (cx-px)^2 + (cy-py)^2 |

    If the determinant is positive and (a, b, c) is counter-clockwise,
    then p is inside the circumcircle.

    Parameters
    ----------
    p : Tensor
        Point to test, shape (2,).
    a, b, c : Tensor
        Triangle vertices, each shape (2,).

    Returns
    -------
    bool
        True if p is strictly inside the circumcircle.
    """
    # Translate so p is at origin
    ax, ay = (a[0] - p[0]).item(), (a[1] - p[1]).item()
    bx, by = (b[0] - p[0]).item(), (b[1] - p[1]).item()
    cx, cy = (c[0] - p[0]).item(), (c[1] - p[1]).item()

    # Compute the determinant
    det = (
        ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
        - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by))
        + (ax * ax + ay * ay) * (bx * cy - by * cx)
    )

    # The sign of the determinant tells us if p is inside
    # We also need to account for the orientation of the triangle
    orient = _orient_2d(a, b, c)

    if orient > 0:
        return det > 0
    elif orient < 0:
        return det < 0
    else:
        # Degenerate triangle
        return False


def _orient_2d(a: Tensor, b: Tensor, c: Tensor) -> float:
    """Compute the orientation of triangle (a, b, c).

    Returns a positive value if the vertices are in counter-clockwise order,
    negative if clockwise, and zero if collinear.

    Parameters
    ----------
    a, b, c : Tensor
        Triangle vertices, each shape (2,).

    Returns
    -------
    float
        Signed area * 2 of the triangle.
    """
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

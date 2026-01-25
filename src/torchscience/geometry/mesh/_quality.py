"""Mesh quality metrics for finite element meshes."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ._mesh import Mesh


def mesh_quality(
    mesh: Mesh,
    metric: str = "aspect_ratio",
) -> Tensor:
    """Compute element quality metrics for a mesh.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    metric : str, optional
        Quality metric to compute. Options:
        - "aspect_ratio": Ratio of longest to shortest edge (1.0 is ideal).
        - "min_angle": Minimum interior angle in degrees.
        Default "aspect_ratio".

    Returns
    -------
    Tensor
        Quality values per element, shape (num_elements,).
        For aspect_ratio: values >= 1.0, lower is better.
        For min_angle: values in degrees, higher is better (up to ideal).

    Examples
    --------
    >>> from torchscience.geometry.mesh import rectangle_mesh, mesh_quality
    >>> mesh = rectangle_mesh(2, 2, [[0.0, 1.0], [0.0, 1.0]])
    >>> quality = mesh_quality(mesh, metric="aspect_ratio")
    >>> quality.shape
    torch.Size([8])
    """
    if metric not in ("aspect_ratio", "min_angle"):
        raise ValueError(
            f"metric must be 'aspect_ratio' or 'min_angle', got '{metric}'"
        )

    element_type = mesh.element_type

    if element_type == "triangle":
        return _triangle_quality(mesh, metric)
    elif element_type == "quad":
        return _quad_quality(mesh, metric)
    else:
        raise ValueError(
            f"mesh_quality not supported for element type '{element_type}'"
        )


def _triangle_quality(mesh: Mesh, metric: str) -> Tensor:
    """Compute quality metric for triangular elements."""
    vertices = mesh.vertices  # (num_vertices, dim)
    elements = mesh.elements  # (num_elements, 3)

    # Get vertex positions for each element
    # Shape: (num_elements, 3, dim)
    v0 = vertices[elements[:, 0]]
    v1 = vertices[elements[:, 1]]
    v2 = vertices[elements[:, 2]]

    # Compute edge vectors
    e0 = v1 - v0  # edge from v0 to v1
    e1 = v2 - v1  # edge from v1 to v2
    e2 = v0 - v2  # edge from v2 to v0

    # Compute edge lengths
    len0 = torch.norm(e0, dim=-1)
    len1 = torch.norm(e1, dim=-1)
    len2 = torch.norm(e2, dim=-1)

    if metric == "aspect_ratio":
        # Stack edge lengths and find min/max
        edge_lengths = torch.stack([len0, len1, len2], dim=-1)
        max_len = edge_lengths.max(dim=-1).values
        min_len = edge_lengths.min(dim=-1).values

        return max_len / min_len

    elif metric == "min_angle":
        # Use law of cosines to compute angles
        # For angle at vertex v0: opposite edge is e1
        # cos(angle) = (len0^2 + len2^2 - len1^2) / (2 * len0 * len2)
        # But we need vectors pointing away from each vertex

        # Angle at v0: between edges (v1-v0) and (v2-v0)
        # Angle at v1: between edges (v0-v1) and (v2-v1)
        # Angle at v2: between edges (v0-v2) and (v1-v2)

        # Using dot product: cos(angle) = (a . b) / (|a| |b|)
        # Vectors from v0: e0 = v1-v0, and (v2-v0) = -e2
        cos_angle0 = torch.sum(e0 * (-e2), dim=-1) / (len0 * len2)

        # Vectors from v1: -e0 = v0-v1, and e1 = v2-v1
        cos_angle1 = torch.sum((-e0) * e1, dim=-1) / (len0 * len1)

        # Vectors from v2: -e1 = v1-v2, and e2 = v0-v2
        cos_angle2 = torch.sum((-e1) * e2, dim=-1) / (len1 * len2)

        # Clamp to avoid numerical issues with acos
        cos_angle0 = torch.clamp(cos_angle0, -1.0, 1.0)
        cos_angle1 = torch.clamp(cos_angle1, -1.0, 1.0)
        cos_angle2 = torch.clamp(cos_angle2, -1.0, 1.0)

        # Convert to angles in degrees
        angle0 = torch.acos(cos_angle0) * (180.0 / math.pi)
        angle1 = torch.acos(cos_angle1) * (180.0 / math.pi)
        angle2 = torch.acos(cos_angle2) * (180.0 / math.pi)

        # Stack and find minimum
        angles = torch.stack([angle0, angle1, angle2], dim=-1)
        return angles.min(dim=-1).values

    # This should never be reached due to validation above
    raise ValueError(f"Unknown metric: {metric}")


def _quad_quality(mesh: Mesh, metric: str) -> Tensor:
    """Compute quality metric for quadrilateral elements."""
    vertices = mesh.vertices  # (num_vertices, dim)
    elements = mesh.elements  # (num_elements, 4)

    # Get vertex positions for each element
    # Vertices are ordered counter-clockwise: v0, v1, v2, v3
    v0 = vertices[elements[:, 0]]
    v1 = vertices[elements[:, 1]]
    v2 = vertices[elements[:, 2]]
    v3 = vertices[elements[:, 3]]

    # Compute edge vectors (counter-clockwise)
    e0 = v1 - v0  # edge from v0 to v1
    e1 = v2 - v1  # edge from v1 to v2
    e2 = v3 - v2  # edge from v2 to v3
    e3 = v0 - v3  # edge from v3 to v0

    # Compute edge lengths
    len0 = torch.norm(e0, dim=-1)
    len1 = torch.norm(e1, dim=-1)
    len2 = torch.norm(e2, dim=-1)
    len3 = torch.norm(e3, dim=-1)

    if metric == "aspect_ratio":
        # Stack edge lengths and find min/max
        edge_lengths = torch.stack([len0, len1, len2, len3], dim=-1)
        max_len = edge_lengths.max(dim=-1).values
        min_len = edge_lengths.min(dim=-1).values

        return max_len / min_len

    elif metric == "min_angle":
        # Compute angles at each vertex using dot product
        # Angle at v0: between edges (v1-v0) and (v3-v0) = e0 and -e3
        cos_angle0 = torch.sum(e0 * (-e3), dim=-1) / (len0 * len3)

        # Angle at v1: between edges (v2-v1) and (v0-v1) = e1 and -e0
        cos_angle1 = torch.sum(e1 * (-e0), dim=-1) / (len1 * len0)

        # Angle at v2: between edges (v3-v2) and (v1-v2) = e2 and -e1
        cos_angle2 = torch.sum(e2 * (-e1), dim=-1) / (len2 * len1)

        # Angle at v3: between edges (v0-v3) and (v2-v3) = e3 and -e2
        cos_angle3 = torch.sum(e3 * (-e2), dim=-1) / (len3 * len2)

        # Clamp to avoid numerical issues with acos
        cos_angle0 = torch.clamp(cos_angle0, -1.0, 1.0)
        cos_angle1 = torch.clamp(cos_angle1, -1.0, 1.0)
        cos_angle2 = torch.clamp(cos_angle2, -1.0, 1.0)
        cos_angle3 = torch.clamp(cos_angle3, -1.0, 1.0)

        # Convert to angles in degrees
        angle0 = torch.acos(cos_angle0) * (180.0 / math.pi)
        angle1 = torch.acos(cos_angle1) * (180.0 / math.pi)
        angle2 = torch.acos(cos_angle2) * (180.0 / math.pi)
        angle3 = torch.acos(cos_angle3) * (180.0 / math.pi)

        # Stack and find minimum
        angles = torch.stack([angle0, angle1, angle2, angle3], dim=-1)
        return angles.min(dim=-1).values

    # This should never be reached due to validation above
    raise ValueError(f"Unknown metric: {metric}")

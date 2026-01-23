"""Line integral computation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._path import Path


def _interpolate_2d(
    field: Tensor,
    points: Tensor,
    spacing: float | Sequence[float],
) -> Tensor:
    """Bilinear interpolation of 2D field at arbitrary points.

    Parameters
    ----------
    field : Tensor
        Field with shape (H, W).
    points : Tensor
        Points with shape (N, 2) in physical coordinates.
    spacing : float or sequence
        Grid spacing to convert to grid indices.
    """
    H, W = field.shape[-2:]

    # Convert points to grid indices
    if isinstance(spacing, (int, float)):
        spacing_list = [spacing, spacing]
    else:
        spacing_list = list(spacing)

    grid_points = points.clone()
    grid_points[:, 0] = points[:, 0] / spacing_list[0]
    grid_points[:, 1] = points[:, 1] / spacing_list[1]

    # Normalize to [-1, 1] for grid_sample
    grid_points[:, 0] = 2 * grid_points[:, 0] / (H - 1) - 1
    grid_points[:, 1] = 2 * grid_points[:, 1] / (W - 1) - 1

    # Reshape field for grid_sample: (1, 1, H, W)
    field_4d = field.unsqueeze(0).unsqueeze(0)

    # Reshape grid for grid_sample: (1, 1, N, 2), note: (x, y) order for grid_sample
    grid = grid_points.flip(-1).unsqueeze(0).unsqueeze(0)

    sampled = F.grid_sample(
        field_4d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return sampled.squeeze()


def _line_integral_impl(
    vector_field: Tensor,
    path: Path,
    spacing: Tensor | Sequence[float] | float = 1.0,
) -> Tensor:
    """Implementation of line integral computation."""
    ndim = vector_field.shape[0]
    points = path.points

    # Ensure points have same dtype as vector_field
    if points.dtype != vector_field.dtype:
        points = points.to(vector_field.dtype)
        path = Path(points=points, closed=path.closed)

    if isinstance(spacing, (int, float)):
        spacing_list = [spacing] * ndim
    elif isinstance(spacing, Tensor):
        spacing_list = spacing.tolist()
    else:
        spacing_list = list(spacing)

    # Get tangent vectors (segment directions)
    tangents = path.tangents  # (N-1, ndim) or (N, ndim) for closed

    # Midpoints of each segment
    midpoints = path.midpoints

    # Interpolate field at midpoints
    if ndim == 2:
        F_x = _interpolate_2d(vector_field[0], midpoints, spacing_list)
        F_y = _interpolate_2d(vector_field[1], midpoints, spacing_list)
        F_at_mid = torch.stack([F_x, F_y], dim=-1)
    else:
        raise NotImplementedError(
            "Only 2D line integrals currently implemented"
        )

    # Dot product F . dl and sum
    integrand = (F_at_mid * tangents).sum(dim=-1)
    result = integrand.sum()

    return result


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def line_integral(
    vector_field: Tensor,
    path: Path,
    spacing: Tensor | Sequence[float] | float = 1.0,
) -> Tensor:
    """Compute line integral along a path.

    Computes the line integral of a vector field F along a discretized path:

    .. math::
        \\int_C \\mathbf{F} \\cdot d\\mathbf{l}

    where the integral is approximated using the midpoint rule.

    Parameters
    ----------
    vector_field : Tensor
        Vector field with shape (ndim, *spatial).
    path : Path
        Discretized path.
    spacing : float or sequence of floats
        Grid spacing.

    Returns
    -------
    Tensor
        Scalar value of the line integral.

    Notes
    -----
    The line integral is approximated using the midpoint rule:

    .. math::
        \\int_C \\mathbf{F} \\cdot d\\mathbf{l} \\approx
        \\sum_i \\mathbf{F}(\\text{mid}_i) \\cdot \\Delta\\mathbf{r}_i

    where :math:`\\text{mid}_i` is the midpoint of segment :math:`i`.

    Examples
    --------
    >>> # Work done by force field along a path
    >>> force_field = torch.randn(2, 32, 32)
    >>> path = Path(points=torch.rand(10, 2))
    >>> work = line_integral(force_field, path, spacing=1/31)
    """
    return _line_integral_impl(vector_field, path, spacing)


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def circulation(
    vector_field: Tensor,
    contour: Path,
    spacing: Tensor | Sequence[float] | float = 1.0,
) -> Tensor:
    """Compute circulation around a closed contour.

    Computes the closed line integral of a vector field F around a contour:

    .. math::
        \\Gamma = \\oint_C \\mathbf{F} \\cdot d\\mathbf{l}

    By Stokes' theorem, this equals the flux of curl through any surface
    bounded by the contour.

    Parameters
    ----------
    vector_field : Tensor
        Vector field with shape (ndim, *spatial).
    contour : Path
        Closed path (contour.closed should be True).
    spacing : float or sequence of floats
        Grid spacing.

    Returns
    -------
    Tensor
        Scalar value of the circulation.

    Notes
    -----
    By Stokes' theorem, circulation equals the flux of curl through
    any surface bounded by the contour:

    .. math::
        \\oint_C \\mathbf{F} \\cdot d\\mathbf{l} =
        \\iint_S (\\nabla \\times \\mathbf{F}) \\cdot d\\mathbf{A}

    Examples
    --------
    >>> # Circulation around a vortex
    >>> velocity = torch.randn(2, 32, 32)
    >>> theta = torch.linspace(0, 2*torch.pi, 50)
    >>> contour = Path(
    ...     points=torch.stack([torch.cos(theta), torch.sin(theta)], dim=1),
    ...     closed=True
    ... )
    >>> gamma = circulation(velocity, contour, spacing=1/31)
    """
    if not contour.closed:
        # Force closed interpretation
        contour = Path(points=contour.points, closed=True)

    return _line_integral_impl(vector_field, contour, spacing)

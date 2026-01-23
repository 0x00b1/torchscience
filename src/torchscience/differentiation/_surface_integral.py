"""Surface integral computation."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._path import Surface


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def surface_integral(
    field: Tensor,
    surface: Surface,
) -> Tensor:
    r"""Compute surface integral over a surface.

    Computes the surface integral of a scalar field f over a surface S:

    .. math::
        \iint_S f \, dA

    where dA is the area element on the surface.

    Parameters
    ----------
    field : Tensor
        Scalar field sampled at surface points with shape (Nu, Nv).
    surface : Surface
        Discretized surface with shape (Nu, Nv, 3).

    Returns
    -------
    Tensor
        Scalar value of the surface integral.

    Notes
    -----
    The surface integral is approximated using the midpoint rule with
    area elements computed from the surface parametrization:

    .. math::
        \iint_S f \, dA \approx \sum_{i,j} f_{ij} \cdot |dA_{ij}|

    where :math:`|dA_{ij}|` is the magnitude of the cross product of
    the tangent vectors at each grid point.

    Examples
    --------
    >>> # Total mass on a surface
    >>> n = 16
    >>> u = torch.linspace(0, 1, n)
    >>> v = torch.linspace(0, 1, n)
    >>> U, V = torch.meshgrid(u, v, indexing="ij")
    >>> points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
    >>> surface = Surface(points=points)
    >>> density = torch.ones(n, n)  # unit density
    >>> total_mass = surface_integral(density, surface)
    """
    # Convert surface area elements to match field dtype for numerical stability
    # Under autocast, field is cast to float32, so area_elements should match
    area_elements = surface.area_elements.to(field.dtype)

    # Integrate: integral f dA = sum f_ij * dA_ij
    integrand = field * area_elements
    result = integrand.sum()

    return result


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def flux(
    vector_field: Tensor,
    surface: Surface,
) -> Tensor:
    r"""Compute flux through a surface.

    Computes the flux of a vector field F through a surface S:

    .. math::
        \iint_S \mathbf{F} \cdot \mathbf{n} \, dA

    where n is the unit normal vector to the surface.

    Parameters
    ----------
    vector_field : Tensor
        Vector field sampled at surface points with shape (3, Nu, Nv).
    surface : Surface
        Discretized surface with shape (Nu, Nv, 3).

    Returns
    -------
    Tensor
        Scalar value of the flux integral.

    Notes
    -----
    The flux measures how much of the vector field passes through
    the surface. By Gauss's divergence theorem:

    .. math::
        \iiint_V \nabla \cdot \mathbf{F} \, dV =
        \oiint_S \mathbf{F} \cdot \mathbf{n} \, dA

    where V is the volume enclosed by the closed surface S.

    The sign of the flux depends on the orientation of the surface
    normals, which is determined by the order of the parametrization.

    Examples
    --------
    >>> # Electric flux through a surface
    >>> n = 16
    >>> u = torch.linspace(0, 1, n)
    >>> v = torch.linspace(0, 1, n)
    >>> U, V = torch.meshgrid(u, v, indexing="ij")
    >>> points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
    >>> surface = Surface(points=points)
    >>> E_field = torch.randn(3, n, n)  # electric field
    >>> electric_flux = flux(E_field, surface)
    """
    # Convert surface normals and area elements to match vector_field dtype
    # Under autocast, vector_field is cast to float32, so these should match
    normals = surface.normals.to(vector_field.dtype)  # (Nu, Nv, 3)
    area_elements = surface.area_elements.to(vector_field.dtype)  # (Nu, Nv)

    # Dot product F . n at each point
    # vector_field is (3, Nu, Nv), normals is (Nu, Nv, 3)
    F_dot_n = (
        vector_field[0] * normals[..., 0]
        + vector_field[1] * normals[..., 1]
        + vector_field[2] * normals[..., 2]
    )

    # Integrate: integral F.n dA
    integrand = F_dot_n * area_elements
    result = integrand.sum()

    return result

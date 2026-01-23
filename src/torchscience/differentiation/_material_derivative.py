"""Material derivative for Lagrangian fluid mechanics."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._advect import advect
from torchscience.differentiation._grid import IrregularMesh, RegularGrid


def _material_derivative_impl(
    field: Tensor,
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    time_derivative: Tensor | None = None,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute material derivative Df/Dt = df/dt + (v . grad)f.

    The material derivative represents the rate of change of a quantity
    following a fluid particle (Lagrangian perspective).

    Parameters
    ----------
    field : Tensor
        Scalar field with shape (*spatial).
    velocity : Tensor
        Velocity field with shape (ndim, *spatial).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions (default: all dims of field).
    accuracy : int, optional
        Finite difference accuracy order. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate". Ignored if grid is provided.
    time_derivative : Tensor, optional
        Time derivative df/dt with shape (*spatial).
        If None, assumed to be zero (steady state).
    grid : RegularGrid or IrregularMesh, optional
        Grid defining spacing and boundary conditions. When provided, overrides
        dx and boundary parameters.

    Returns
    -------
    Tensor
        Material derivative Df/Dt with shape (*spatial).

    Notes
    -----
    The material derivative combines:
    - Local (Eulerian) rate of change: df/dt
    - Convective rate of change: (v . grad)f

    For a steady flow (df/dt = 0), Df/Dt = (v . grad)f.
    For a stationary fluid (v = 0), Df/Dt = df/dt.

    Examples
    --------
    >>> # Temperature change following a fluid particle
    >>> temperature = torch.randn(32, 32)
    >>> velocity = torch.randn(2, 32, 32)
    >>> dT_dt = torch.randn(32, 32)  # Local heating rate
    >>> DT_Dt = material_derivative(temperature, velocity,
    ...                             time_derivative=dT_dt, dx=0.1)
    """
    # Compute advection term (v . grad)f
    advection = advect(
        field,
        velocity,
        dx=dx,
        dim=dim,
        accuracy=accuracy,
        boundary=boundary,
        grid=grid,
    )

    # Add time derivative if provided
    if time_derivative is not None:
        return time_derivative + advection
    else:
        return advection


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def material_derivative(
    field: Tensor,
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    time_derivative: Tensor | None = None,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute material derivative Df/Dt = df/dt + (v . grad)f.

    The material derivative (also called substantial or Lagrangian derivative)
    represents the rate of change of a quantity following a fluid particle.

    Parameters
    ----------
    field : Tensor
        Scalar field with shape (*spatial).
    velocity : Tensor
        Velocity field with shape (ndim, *spatial).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions (default: all dims of field).
    accuracy : int, optional
        Finite difference accuracy order. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate". Ignored if grid is provided.
    time_derivative : Tensor, optional
        Time derivative df/dt with shape (*spatial).
        If None, assumed to be zero (steady state).
    grid : RegularGrid or IrregularMesh, optional
        Grid defining spacing and boundary conditions. When provided, overrides
        dx and boundary parameters.

    Returns
    -------
    Tensor
        Material derivative Df/Dt with shape (*spatial).

    Notes
    -----
    The material derivative combines:

    .. math::

        \\frac{Df}{Dt} = \\frac{\\partial f}{\\partial t} + (\\mathbf{v} \\cdot \\nabla) f

    where:

    - :math:`\\frac{\\partial f}{\\partial t}` is the local (Eulerian) rate of change
    - :math:`(\\mathbf{v} \\cdot \\nabla) f` is the convective rate of change

    For a steady flow (:math:`\\frac{\\partial f}{\\partial t} = 0`), the material
    derivative equals the advection term:

    .. math::

        \\frac{Df}{Dt} = (\\mathbf{v} \\cdot \\nabla) f

    For a stationary fluid (:math:`\\mathbf{v} = 0`), the material derivative
    equals the local time derivative:

    .. math::

        \\frac{Df}{Dt} = \\frac{\\partial f}{\\partial t}

    Examples
    --------
    >>> import torch
    >>> from torchscience.differentiation import material_derivative
    >>>
    >>> # Temperature change following a fluid particle
    >>> temperature = torch.randn(32, 32)
    >>> velocity = torch.randn(2, 32, 32)
    >>> dT_dt = torch.randn(32, 32)  # Local heating rate
    >>> DT_Dt = material_derivative(temperature, velocity,
    ...                             time_derivative=dT_dt, dx=0.1)
    >>>
    >>> # Steady flow (no explicit time dependence)
    >>> DT_Dt_steady = material_derivative(temperature, velocity, dx=0.1)
    """
    return _material_derivative_impl(
        field,
        velocity,
        dx,
        dim,
        accuracy,
        boundary,
        time_derivative=time_derivative,
        grid=grid,
    )

"""Helicity computation for turbulence analysis."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._vorticity import vorticity


def _helicity_impl(
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    integrated: bool = False,
) -> Tensor:
    """Compute helicity of a velocity field.

    Helicity is v . omega where omega is vorticity.

    Parameters
    ----------
    velocity : Tensor
        Velocity field with shape (ndim, *spatial).
    dx : float or tuple of floats
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
    dim : tuple of int, optional
        Spatial dimensions.
    accuracy : int
        Finite difference accuracy order.
    boundary : str
        Boundary condition.
    integrated : bool
        If True, return total integrated helicity (scalar).
        If False, return helicity field.

    Returns
    -------
    Tensor
        Helicity field (*spatial) or scalar if integrated.
    """
    ndim = velocity.shape[0]

    if ndim == 2:
        # In 2D, vorticity is perpendicular to the velocity plane.
        # The vorticity vector is (0, 0, omega_z) while velocity is (v_x, v_y, 0).
        # Therefore v . omega = 0 always in 2D.
        spatial_shape = velocity.shape[1:]
        if integrated:
            return torch.tensor(
                0.0, device=velocity.device, dtype=velocity.dtype
            )
        return torch.zeros(
            spatial_shape, device=velocity.device, dtype=velocity.dtype
        )

    # 3D case: compute vorticity and dot with velocity
    omega = vorticity(
        velocity, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary
    )

    # Dot product v . omega (sum over component dimension)
    h = (velocity * omega).sum(dim=0)

    if integrated:
        # Integrate over domain
        if isinstance(dx, (int, float)):
            dV = float(dx) ** ndim
        elif isinstance(dx, Tensor):
            dV = dx.prod().item()
        else:
            dV = 1.0
            for s in dx:
                dV *= float(s)
        return h.sum() * dV

    return h


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def helicity(
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    integrated: bool = False,
) -> Tensor:
    r"""Compute helicity of a velocity field.

    Helicity is v . omega where omega is vorticity (curl of velocity).

    Parameters
    ----------
    velocity : Tensor
        Velocity field with shape (ndim, *spatial) where ndim is 2 or 3.
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions (default: last ndim dims after component dimension).
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".
    integrated : bool, optional
        If True, return total integrated helicity (scalar).
        If False, return helicity field. Default is False.

    Returns
    -------
    Tensor
        Helicity field (*spatial) or scalar if integrated.

    Notes
    -----
    Helicity measures the degree to which vortex lines are linked or knotted.
    It is a conserved quantity in inviscid, barotropic flows.

    In 2D, helicity is always zero because the vorticity vector is perpendicular
    to the velocity plane.

    The helicity is defined as:

    .. math::

        \mathcal{H} = \mathbf{v} \cdot \boldsymbol{\omega}

    where :math:`\boldsymbol{\omega} = \nabla \times \mathbf{v}` is the vorticity.

    Positive helicity indicates right-handed helical structures (vortex lines
    tend to twist in the same sense as they wind around each other).
    Negative helicity indicates left-handed helical structures.

    For Beltrami flows (including ABC flows), the velocity and vorticity
    are parallel everywhere, maximizing the helicity density.

    Examples
    --------
    >>> velocity = torch.randn(3, 16, 16, 16)
    >>> h = helicity(velocity, dx=0.1)
    >>> h.shape
    torch.Size([16, 16, 16])

    >>> # Integrated helicity (scalar)
    >>> total_h = helicity(velocity, dx=0.1, integrated=True)
    >>> total_h.ndim
    0
    """
    return _helicity_impl(
        velocity, dx, dim, accuracy, boundary, integrated=integrated
    )

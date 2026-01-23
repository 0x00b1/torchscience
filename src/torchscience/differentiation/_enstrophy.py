"""Enstrophy computation for turbulence analysis."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._vorticity import vorticity


def _enstrophy_impl(
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    integrated: bool = False,
) -> Tensor:
    """Compute enstrophy of a velocity field.

    Enstrophy is 1/2 |omega|^2 where omega is vorticity.

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
        If True, return total integrated enstrophy (scalar).
        If False, return enstrophy field.

    Returns
    -------
    Tensor
        Enstrophy field (*spatial) or scalar if integrated.
    """
    omega = vorticity(
        velocity, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary
    )

    ndim = velocity.shape[0]

    if ndim == 2:
        # Scalar vorticity
        ens = 0.5 * omega**2
    else:
        # Vector vorticity - sum of squared components
        ens = 0.5 * (omega**2).sum(dim=0)

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
        return ens.sum() * dV

    return ens


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def enstrophy(
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    integrated: bool = False,
) -> Tensor:
    """Compute enstrophy of a velocity field.

    Enstrophy is 1/2 |omega|^2 where omega is vorticity.

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
        If True, return total integrated enstrophy (scalar).
        If False, return enstrophy field. Default is False.

    Returns
    -------
    Tensor
        Enstrophy field (*spatial) or scalar if integrated.

    Notes
    -----
    Enstrophy is a measure of rotational intensity in a flow.
    In 2D turbulence, enstrophy is conserved in inviscid flows.

    The enstrophy is defined as:

    .. math::

        \\mathcal{E} = \\frac{1}{2} |\\omega|^2

    where :math:`\\omega` is the vorticity (curl of velocity).

    Examples
    --------
    >>> velocity = torch.randn(2, 32, 32)
    >>> ens = enstrophy(velocity, dx=0.1)
    >>> ens.shape
    torch.Size([32, 32])

    >>> # Integrated enstrophy (scalar)
    >>> total_ens = enstrophy(velocity, dx=0.1, integrated=True)
    >>> total_ens.ndim
    0
    """
    return _enstrophy_impl(
        velocity, dx, dim, accuracy, boundary, integrated=integrated
    )

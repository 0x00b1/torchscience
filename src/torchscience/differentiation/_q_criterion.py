"""Q-criterion for vortex identification."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._jacobian import jacobian


def _q_criterion_impl(
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute Q-criterion for vortex identification.

    Q = 1/2 (|Omega|^2 - |S|^2)

    where Omega is the rotation rate tensor and S is the strain rate tensor.
    Positive Q indicates rotation dominates strain (vortex cores).

    Parameters
    ----------
    velocity : Tensor
        Velocity field with shape (3, *spatial).
    dx : float or tuple of floats
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
    dim : tuple of int, optional
        Spatial dimensions.
    accuracy : int
        Finite difference accuracy order.
    boundary : str
        Boundary condition.

    Returns
    -------
    Tensor
        Q-criterion field (*spatial).
    """
    ndim = velocity.shape[0]
    if ndim != 3:
        raise ValueError(
            f"Q-criterion requires 3D velocity field, got {ndim}D"
        )

    # Compute velocity gradient tensor: J_ij = dv_i / dx_j
    # jacobian returns shape (3, 3, *spatial)
    J = jacobian(
        velocity, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary
    )

    # Strain rate tensor: S = 1/2 (J + J^T)
    S = 0.5 * (J + J.transpose(0, 1))

    # Rotation rate tensor: Omega = 1/2 (J - J^T)
    Omega = 0.5 * (J - J.transpose(0, 1))

    # Frobenius norms squared
    # |S|^2 = sum_ij S_ij^2
    S_norm_sq = (S**2).sum(dim=(0, 1))
    Omega_norm_sq = (Omega**2).sum(dim=(0, 1))

    # Q = 1/2 (|Omega|^2 - |S|^2)
    Q = 0.5 * (Omega_norm_sq - S_norm_sq)

    return Q


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def q_criterion(
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    r"""Compute Q-criterion for vortex identification.

    Q = 1/2 (|Omega|^2 - |S|^2)

    where Omega is the rotation rate tensor and S is the strain rate tensor.
    Positive Q indicates rotation dominates strain (vortex cores).

    Parameters
    ----------
    velocity : Tensor
        Velocity field with shape (3, *spatial).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions (default: last 3 dims after component dimension).
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Q-criterion field (*spatial).

    Notes
    -----
    The Q-criterion is commonly used in fluid dynamics to identify
    vortex structures. Isosurfaces of Q > 0 often correspond to vortex
    boundaries.

    The velocity gradient tensor is decomposed as:

    .. math::

        \frac{\partial v_i}{\partial x_j} = S_{ij} + \Omega_{ij}

    where:
        - :math:`S_{ij} = \frac{1}{2}\left(\frac{\partial v_i}{\partial x_j} +
          \frac{\partial v_j}{\partial x_i}\right)` (symmetric, strain rate)
        - :math:`\Omega_{ij} = \frac{1}{2}\left(\frac{\partial v_i}{\partial x_j} -
          \frac{\partial v_j}{\partial x_i}\right)` (antisymmetric, rotation rate)

    The Q-criterion is defined as:

    .. math::

        Q = \frac{1}{2}\left(|\Omega|^2 - |S|^2\right)

    where :math:`|\cdot|` denotes the Frobenius norm.

    Positive Q indicates regions where rotation dominates strain (vortex cores).
    Negative Q indicates regions where strain dominates rotation.

    Examples
    --------
    >>> # Identify vortex cores
    >>> velocity = torch.randn(3, 32, 32, 32)
    >>> Q = q_criterion(velocity, dx=0.1)
    >>> vortex_mask = Q > 0
    """
    return _q_criterion_impl(velocity, dx, dim, accuracy, boundary)

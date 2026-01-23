"""Twist (screw) representation for SE(3) Lie algebra.

This module provides the Twist tensorclass and associated operations for
representing 6-DOF spatial velocities (se(3) Lie algebra elements).

A twist combines angular velocity (omega) and linear velocity (v) into a
single 6-dimensional vector. It represents the instantaneous velocity of
a rigid body in 3D space.
"""

from __future__ import annotations

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor


@tensorclass
class Twist:
    """6-DOF velocity/screw representation (se(3) Lie algebra element).

    A twist represents the instantaneous spatial velocity of a rigid body,
    combining angular and linear velocities.

    Attributes
    ----------
    angular : Tensor
        Angular velocity component (omega), shape (..., 3).
        Represents the rotation axis scaled by angular speed.
    linear : Tensor
        Linear velocity component (v), shape (..., 3).
        Represents the velocity of the body-fixed origin.

    Notes
    -----
    The twist is stored as a pair (angular, linear) rather than a single
    6D vector for clarity. Use `twist_to_vector` to convert to vector form.

    The "hat" operator maps a twist to a 4x4 matrix in se(3):

    .. math::

        \\hat{\\xi} = \\begin{bmatrix}
            [\\omega]_\\times & v \\\\
            0 & 0
        \\end{bmatrix}

    where :math:`[\\omega]_\\times` is the 3x3 skew-symmetric matrix.

    Examples
    --------
    Pure rotation (angular velocity only):
        Twist(angular=torch.tensor([0.0, 0.0, 1.0]), linear=torch.zeros(3))

    Pure translation (linear velocity only):
        Twist(angular=torch.zeros(3), linear=torch.tensor([1.0, 0.0, 0.0]))

    Screw motion (combined):
        Twist(angular=torch.tensor([0.0, 0.0, 0.5]),
              linear=torch.tensor([1.0, 0.0, 0.0]))
    """

    angular: Tensor
    linear: Tensor


def twist(angular: Tensor, linear: Tensor) -> Twist:
    """Create a Twist from angular and linear velocity components.

    Parameters
    ----------
    angular : Tensor
        Angular velocity (omega), shape (..., 3).
    linear : Tensor
        Linear velocity (v), shape (..., 3).

    Returns
    -------
    Twist
        Twist instance with the given components.

    Raises
    ------
    ValueError
        If angular or linear does not have last dimension 3.
        If angular and linear batch shapes cannot broadcast.

    Examples
    --------
    >>> omega = torch.tensor([0.0, 0.0, 1.0])
    >>> v = torch.tensor([1.0, 0.0, 0.0])
    >>> t = twist(omega, v)
    >>> t.angular
    tensor([0., 0., 1.])
    """
    if angular.shape[-1] != 3:
        raise ValueError(
            f"twist: angular must have last dimension 3, got {angular.shape[-1]}"
        )
    if linear.shape[-1] != 3:
        raise ValueError(
            f"twist: linear must have last dimension 3, got {linear.shape[-1]}"
        )

    # Check that batch shapes can broadcast
    angular_batch = angular.shape[:-1]
    linear_batch = linear.shape[:-1]
    try:
        torch.broadcast_shapes(angular_batch, linear_batch)
    except RuntimeError:
        raise ValueError(
            f"twist: angular batch shape {angular_batch} and "
            f"linear batch shape {linear_batch} cannot broadcast"
        )

    return Twist(angular=angular, linear=linear)


def twist_from_vector(vector: Tensor) -> Twist:
    """Create a Twist from a 6D vector [angular; linear].

    Parameters
    ----------
    vector : Tensor
        6D twist vector, shape (..., 6). First 3 components are angular
        velocity, last 3 are linear velocity.

    Returns
    -------
    Twist
        Twist instance.

    Raises
    ------
    ValueError
        If vector does not have last dimension 6.

    Examples
    --------
    >>> vec = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    >>> t = twist_from_vector(vec)
    >>> t.angular
    tensor([0., 0., 1.])
    >>> t.linear
    tensor([1., 0., 0.])
    """
    if vector.shape[-1] != 6:
        raise ValueError(
            f"twist_from_vector: vector must have last dimension 6, "
            f"got {vector.shape[-1]}"
        )

    angular = vector[..., :3]
    linear = vector[..., 3:]
    return Twist(angular=angular, linear=linear)


def twist_to_vector(t: Twist) -> Tensor:
    """Convert a Twist to a 6D vector [angular; linear].

    Parameters
    ----------
    t : Twist
        Twist to convert.

    Returns
    -------
    Tensor
        6D twist vector, shape (..., 6).

    Examples
    --------
    >>> omega = torch.tensor([0.0, 0.0, 1.0])
    >>> v = torch.tensor([1.0, 0.0, 0.0])
    >>> t = twist(omega, v)
    >>> twist_to_vector(t)
    tensor([0., 0., 1., 1., 0., 0.])
    """
    return torch.cat([t.angular, t.linear], dim=-1)


def _skew_symmetric_3x3(omega: Tensor) -> Tensor:
    """Compute the skew-symmetric matrix [omega]_x from a 3-vector.

    Parameters
    ----------
    omega : Tensor
        3-vector, shape (..., 3).

    Returns
    -------
    Tensor
        Skew-symmetric matrix, shape (..., 3, 3).
    """
    omega_x = omega[..., 0]
    omega_y = omega[..., 1]
    omega_z = omega[..., 2]

    zeros = torch.zeros_like(omega_x)

    row0 = torch.stack([zeros, -omega_z, omega_y], dim=-1)
    row1 = torch.stack([omega_z, zeros, -omega_x], dim=-1)
    row2 = torch.stack([-omega_y, omega_x, zeros], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def twist_to_matrix(t: Twist) -> Tensor:
    """Convert a Twist to its 4x4 matrix representation (hat operator).

    The "hat" operator maps a twist (se(3) element) to a 4x4 matrix:

    .. math::

        \\hat{\\xi} = \\begin{bmatrix}
            [\\omega]_\\times & v \\\\
            0 & 0
        \\end{bmatrix}

    where :math:`[\\omega]_\\times` is the 3x3 skew-symmetric matrix of the
    angular velocity.

    Parameters
    ----------
    t : Twist
        Twist to convert.

    Returns
    -------
    Tensor
        4x4 se(3) matrix, shape (..., 4, 4).

    Examples
    --------
    >>> omega = torch.tensor([1.0, 0.0, 0.0])
    >>> v = torch.tensor([0.0, 1.0, 0.0])
    >>> mat = twist_to_matrix(twist(omega, v))
    >>> mat.shape
    torch.Size([4, 4])
    """
    batch_shape = t.angular.shape[:-1]
    device = t.angular.device
    dtype = t.angular.dtype

    # Build the 4x4 matrix
    matrix = torch.zeros((*batch_shape, 4, 4), device=device, dtype=dtype)

    # Upper-left 3x3: skew-symmetric matrix of angular velocity
    matrix[..., :3, :3] = _skew_symmetric_3x3(t.angular)

    # Upper-right 3x1: linear velocity
    matrix[..., :3, 3] = t.linear

    # Last row is zeros (already initialized)

    return matrix


def matrix_to_twist(matrix: Tensor) -> Twist:
    """Convert a 4x4 se(3) matrix to a Twist (vee operator).

    The "vee" operator is the inverse of the "hat" operator, extracting
    the twist vector from its matrix representation.

    Parameters
    ----------
    matrix : Tensor
        4x4 se(3) matrix, shape (..., 4, 4).

    Returns
    -------
    Twist
        Twist extracted from the matrix.

    Raises
    ------
    ValueError
        If matrix does not have shape (..., 4, 4).

    Examples
    --------
    >>> mat = torch.zeros(4, 4)
    >>> mat[0, 1] = -1.0  # omega_z = 1
    >>> mat[1, 0] = 1.0
    >>> mat[0, 3] = 2.0   # v_x = 2
    >>> t = matrix_to_twist(mat)
    >>> t.angular
    tensor([0., 0., 1.])
    >>> t.linear[0]
    tensor(2.)
    """
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(
            f"matrix_to_twist: matrix must have shape (..., 4, 4), "
            f"got shape {matrix.shape}"
        )

    # Extract angular velocity from skew-symmetric part
    # [omega]_x has: omega_x = mat[2,1], omega_y = mat[0,2], omega_z = mat[1,0]
    angular = torch.stack(
        [matrix[..., 2, 1], matrix[..., 0, 2], matrix[..., 1, 0]], dim=-1
    )

    # Extract linear velocity from last column
    linear = matrix[..., :3, 3]

    return Twist(angular=angular, linear=linear)


def twist_apply(t: Twist, point: Tensor, dt: Tensor) -> Tensor:
    """Apply a twist to a point for a given time duration.

    Computes the new position of a point after moving with the given
    twist velocity for time dt:

    .. math::

        p' = \\exp(dt \\cdot \\xi) \\cdot p

    Parameters
    ----------
    t : Twist
        Twist (spatial velocity).
    point : Tensor
        3D point(s), shape (..., 3).
    dt : Tensor
        Time duration, scalar or broadcastable shape.

    Returns
    -------
    Tensor
        Transformed point(s), shape (..., 3).

    Notes
    -----
    This is equivalent to computing se3_exp(dt * twist) and applying
    the resulting rigid transform to the point.

    Examples
    --------
    Pure translation:

    >>> omega = torch.zeros(3)
    >>> v = torch.tensor([1.0, 0.0, 0.0])
    >>> t = twist(omega, v)
    >>> point = torch.zeros(3)
    >>> twist_apply(t, point, torch.tensor(2.0))
    tensor([2., 0., 0.])
    """
    # Import here to avoid circular imports
    from torchscience.geometry.transform._exponential_map import se3_exp
    from torchscience.geometry.transform._rigid_transform import (
        rigid_transform_apply,
    )

    # Handle broadcasting of dt
    dt_expanded = dt
    while dt_expanded.dim() < t.angular.dim():
        dt_expanded = dt_expanded.unsqueeze(-1)

    # Scale the twist by dt
    scaled_angular = t.angular * dt_expanded
    scaled_linear = t.linear * dt_expanded
    scaled_twist = Twist(angular=scaled_angular, linear=scaled_linear)

    # Compute the transformation
    transform = se3_exp(scaled_twist)

    # Apply to point
    return rigid_transform_apply(transform, point)

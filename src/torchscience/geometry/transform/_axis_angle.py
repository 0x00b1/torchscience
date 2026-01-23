"""Axis-angle rotation representation and conversions."""

from __future__ import annotations

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.geometry.transform._quaternion import (
    Quaternion,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


@tensorclass
class AxisAngle:
    """Axis-angle rotation representation.

    Represents a 3D rotation as a unit axis and angle in radians.

    Attributes
    ----------
    axis : Tensor
        Unit rotation axis, shape (..., 3).
    angle : Tensor
        Rotation angle in radians, shape (...).

    Examples
    --------
    Identity rotation (zero angle):
        AxisAngle(axis=torch.tensor([0.0, 0.0, 1.0]), angle=torch.tensor(0.0))

    90-degree rotation around z-axis:
        AxisAngle(axis=torch.tensor([0.0, 0.0, 1.0]), angle=torch.tensor(math.pi / 2))

    Batch of axis-angles:
        AxisAngle(axis=torch.randn(100, 3), angle=torch.randn(100))
    """

    axis: Tensor
    angle: Tensor


def axis_angle(axis: Tensor, angle: Tensor) -> AxisAngle:
    """Create axis-angle from axis and angle tensors.

    Parameters
    ----------
    axis : Tensor
        Unit rotation axis, shape (..., 3).
    angle : Tensor
        Rotation angle in radians, shape (...).

    Returns
    -------
    AxisAngle
        AxisAngle instance.

    Raises
    ------
    ValueError
        If axis does not have last dimension 3.

    Examples
    --------
    >>> aa = axis_angle(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(math.pi / 2))
    >>> aa.axis
    tensor([0., 0., 1.])
    >>> aa.angle
    tensor(1.5708)
    """
    if axis.shape[-1] != 3:
        raise ValueError(
            f"axis_angle: axis must have last dimension 3, got {axis.shape[-1]}"
        )
    return AxisAngle(axis=axis, angle=angle)


def axis_angle_to_quaternion(aa: AxisAngle) -> Quaternion:
    """Convert axis-angle to quaternion.

    The quaternion is computed as:

    .. math::

        q = [\\cos(\\theta/2), \\sin(\\theta/2) \\cdot \\text{axis}]

    where :math:`\\theta` is the rotation angle.

    Parameters
    ----------
    aa : AxisAngle
        Axis-angle representation with axis shape (..., 3) and angle shape (...).

    Returns
    -------
    Quaternion
        Unit quaternion in wxyz convention, shape (..., 4).

    Notes
    -----
    - The output quaternion is always a unit quaternion.
    - The axis should be a unit vector for correct results.

    See Also
    --------
    quaternion_to_axis_angle : Inverse conversion.

    Examples
    --------
    Identity rotation (zero angle):

    >>> aa = axis_angle(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.0))
    >>> axis_angle_to_quaternion(aa).wxyz
    tensor([1., 0., 0., 0.])

    90-degree rotation around z-axis:

    >>> import math
    >>> aa = axis_angle(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(math.pi / 2))
    >>> q = axis_angle_to_quaternion(aa)
    >>> q.wxyz
    tensor([0.7071, 0.0000, 0.0000, 0.7071])
    """
    half_angle = aa.angle / 2
    # Add dimension for broadcasting: angle from (...) to (..., 1)
    half_angle_expanded = half_angle.unsqueeze(-1)

    w = torch.cos(half_angle)
    xyz = torch.sin(half_angle_expanded) * aa.axis

    wxyz = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
    return Quaternion(wxyz=wxyz)


def quaternion_to_axis_angle(q: Quaternion) -> AxisAngle:
    """Convert quaternion to axis-angle.

    Parameters
    ----------
    q : Quaternion
        Unit quaternion in wxyz convention, shape (..., 4).

    Returns
    -------
    AxisAngle
        Axis-angle representation with axis shape (..., 3) and angle shape (...).

    Notes
    -----
    - For identity quaternions (angle = 0), the axis is not well-defined.
      In this case, we return [0, 0, 1] as a default axis.
    - The angle is computed as :math:`2 \\arccos(|w|)` and is always positive.
    - The axis direction is determined by the sign of w.

    See Also
    --------
    axis_angle_to_quaternion : Inverse conversion.

    Examples
    --------
    Identity quaternion:

    >>> from torchscience.geometry.transform import quaternion
    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> aa = quaternion_to_axis_angle(q)
    >>> aa.angle
    tensor(0.)

    90-degree rotation around z-axis:

    >>> import math
    >>> q = quaternion(torch.tensor([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)]))
    >>> aa = quaternion_to_axis_angle(q)
    >>> aa.axis
    tensor([0., 0., 1.])
    >>> aa.angle
    tensor(1.5708)
    """
    wxyz = q.wxyz
    w = wxyz[..., 0]
    xyz = wxyz[..., 1:]

    # Compute the norm of the xyz part (sin(angle/2))
    sin_half_angle = torch.linalg.norm(xyz, dim=-1)

    # Compute angle using atan2 for numerical stability
    # angle = 2 * atan2(sin(angle/2), cos(angle/2)) = 2 * atan2(norm(xyz), w)
    angle = 2 * torch.atan2(sin_half_angle, w)

    # Compute axis: normalize xyz
    # Handle the case where sin_half_angle is zero (identity rotation)
    # Use a small epsilon to avoid division by zero
    eps = torch.finfo(xyz.dtype).eps
    sin_half_angle_safe = torch.clamp(sin_half_angle, min=eps)

    axis = xyz / sin_half_angle_safe.unsqueeze(-1)

    # For identity rotations (very small sin_half_angle), use default axis [0, 0, 1]
    is_identity = sin_half_angle < 1e-6
    if is_identity.any():
        default_axis = torch.zeros_like(axis)
        default_axis[..., 2] = 1.0
        axis = torch.where(is_identity.unsqueeze(-1), default_axis, axis)

    return AxisAngle(axis=axis, angle=angle)


def axis_angle_to_matrix(aa: AxisAngle) -> Tensor:
    """Convert axis-angle to 3x3 rotation matrix.

    Converts via quaternion for numerical stability and consistency.

    Parameters
    ----------
    aa : AxisAngle
        Axis-angle representation with axis shape (..., 3) and angle shape (...).

    Returns
    -------
    Tensor
        Rotation matrix, shape (..., 3, 3).

    Notes
    -----
    - The output matrix is orthogonal with determinant 1.
    - Uses the quaternion intermediate representation for numerical stability.

    See Also
    --------
    matrix_to_axis_angle : Inverse conversion.
    axis_angle_to_quaternion : Convert to quaternion.

    Examples
    --------
    Identity rotation (zero angle):

    >>> aa = axis_angle(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.0))
    >>> axis_angle_to_matrix(aa)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])

    90-degree rotation around z-axis:

    >>> import math
    >>> aa = axis_angle(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(math.pi / 2))
    >>> axis_angle_to_matrix(aa)
    tensor([[ 0., -1.,  0.],
            [ 1.,  0.,  0.],
            [ 0.,  0.,  1.]])
    """
    q = axis_angle_to_quaternion(aa)
    return quaternion_to_matrix(q)


def matrix_to_axis_angle(matrix: Tensor) -> AxisAngle:
    """Convert 3x3 rotation matrix to axis-angle.

    Converts via quaternion for numerical stability.

    Parameters
    ----------
    matrix : Tensor
        Rotation matrix, shape (..., 3, 3).

    Returns
    -------
    AxisAngle
        Axis-angle representation with axis shape (..., 3) and angle shape (...).

    Notes
    -----
    - For valid rotation matrices, the result represents the same rotation.
    - Uses quaternion as intermediate representation for numerical stability.

    See Also
    --------
    axis_angle_to_matrix : Inverse conversion.
    matrix_to_quaternion : Convert to quaternion.

    Examples
    --------
    Identity matrix:

    >>> R = torch.eye(3)
    >>> aa = matrix_to_axis_angle(R)
    >>> aa.angle
    tensor(0.)

    Round-trip conversion:

    >>> import math
    >>> aa = axis_angle(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(math.pi / 2))
    >>> R = axis_angle_to_matrix(aa)
    >>> aa_back = matrix_to_axis_angle(R)
    >>> torch.allclose(aa_back.angle, aa.angle)
    True
    """
    q = matrix_to_quaternion(matrix)
    return quaternion_to_axis_angle(q)

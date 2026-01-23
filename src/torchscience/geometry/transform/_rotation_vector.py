"""Rotation vector (Rodrigues) representation and conversions."""

from __future__ import annotations

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.geometry.transform._axis_angle import (
    AxisAngle,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    quaternion_to_axis_angle,
)
from torchscience.geometry.transform._quaternion import Quaternion


@tensorclass
class RotationVector:
    """Rotation vector (Rodrigues) representation.

    Represents a 3D rotation as a single vector where the direction is the
    rotation axis and the magnitude is the rotation angle in radians.

    The rotation vector is defined as:

    .. math::

        \\mathbf{v} = \\theta \\cdot \\hat{\\mathbf{k}}

    where :math:`\\theta` is the rotation angle and :math:`\\hat{\\mathbf{k}}`
    is the unit rotation axis.

    Attributes
    ----------
    vector : Tensor
        Rotation vector (axis * angle), shape (..., 3).

    Examples
    --------
    Identity rotation (zero vector):
        RotationVector(vector=torch.tensor([0.0, 0.0, 0.0]))

    90-degree rotation around z-axis:
        RotationVector(vector=torch.tensor([0.0, 0.0, math.pi / 2]))

    Batch of rotation vectors:
        RotationVector(vector=torch.randn(100, 3))

    Notes
    -----
    The rotation vector representation is also known as:
    - Rodrigues vector
    - Axis-angle vector
    - Exponential coordinates

    This is a compact 3-parameter representation commonly used in computer
    vision (e.g., OpenCV uses this representation).
    """

    vector: Tensor


def rotation_vector(vector: Tensor) -> RotationVector:
    """Create rotation vector from a tensor.

    Parameters
    ----------
    vector : Tensor
        Rotation vector (axis * angle), shape (..., 3).

    Returns
    -------
    RotationVector
        RotationVector instance.

    Raises
    ------
    ValueError
        If vector does not have last dimension 3.

    Examples
    --------
    >>> rv = rotation_vector(torch.tensor([0.0, 0.0, math.pi / 2]))
    >>> rv.vector
    tensor([0.0000, 0.0000, 1.5708])
    """
    if vector.shape[-1] != 3:
        raise ValueError(
            f"rotation_vector: vector must have last dimension 3, got {vector.shape[-1]}"
        )
    return RotationVector(vector=vector)


def rotation_vector_to_quaternion(rv: RotationVector) -> Quaternion:
    """Convert rotation vector to quaternion.

    The quaternion is computed via axis-angle representation.

    Parameters
    ----------
    rv : RotationVector
        Rotation vector with shape (..., 3).

    Returns
    -------
    Quaternion
        Unit quaternion in wxyz convention, shape (..., 4).

    Notes
    -----
    - The output quaternion is always a unit quaternion.
    - Zero rotation vector gives identity quaternion [1, 0, 0, 0].

    See Also
    --------
    quaternion_to_rotation_vector : Inverse conversion.

    Examples
    --------
    Identity rotation (zero vector):

    >>> rv = rotation_vector(torch.tensor([0.0, 0.0, 0.0]))
    >>> rotation_vector_to_quaternion(rv).wxyz
    tensor([1., 0., 0., 0.])

    90-degree rotation around z-axis:

    >>> import math
    >>> rv = rotation_vector(torch.tensor([0.0, 0.0, math.pi / 2]))
    >>> q = rotation_vector_to_quaternion(rv)
    >>> q.wxyz
    tensor([0.7071, 0.0000, 0.0000, 0.7071])
    """
    # Extract angle (norm of vector) and axis (normalized vector)
    vector = rv.vector
    angle = torch.linalg.norm(vector, dim=-1)

    # Handle zero rotation case
    eps = torch.finfo(vector.dtype).eps
    angle_safe = torch.clamp(angle, min=eps)
    axis = vector / angle_safe.unsqueeze(-1)

    # For zero rotations, the axis doesn't matter (set to z-axis)
    is_zero = angle < 1e-10
    if is_zero.any():
        default_axis = torch.zeros_like(axis)
        default_axis[..., 2] = 1.0
        axis = torch.where(is_zero.unsqueeze(-1), default_axis, axis)

    # Create axis-angle and convert to quaternion
    aa = AxisAngle(axis=axis, angle=angle)
    return axis_angle_to_quaternion(aa)


def quaternion_to_rotation_vector(q: Quaternion) -> RotationVector:
    """Convert quaternion to rotation vector.

    Parameters
    ----------
    q : Quaternion
        Unit quaternion in wxyz convention, shape (..., 4).

    Returns
    -------
    RotationVector
        Rotation vector with shape (..., 3).

    Notes
    -----
    - For identity quaternions (angle = 0), the rotation vector is [0, 0, 0].
    - The rotation vector is computed as axis * angle.

    See Also
    --------
    rotation_vector_to_quaternion : Inverse conversion.

    Examples
    --------
    Identity quaternion:

    >>> from torchscience.geometry.transform import quaternion
    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> rv = quaternion_to_rotation_vector(q)
    >>> rv.vector
    tensor([0., 0., 0.])

    90-degree rotation around z-axis:

    >>> import math
    >>> q = quaternion(torch.tensor([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)]))
    >>> rv = quaternion_to_rotation_vector(q)
    >>> rv.vector
    tensor([0.0000, 0.0000, 1.5708])
    """
    aa = quaternion_to_axis_angle(q)

    # Rotation vector = axis * angle
    angle = aa.angle.unsqueeze(-1)
    vector = aa.axis * angle

    return RotationVector(vector=vector)


def rotation_vector_to_matrix(rv: RotationVector) -> Tensor:
    """Convert rotation vector to 3x3 rotation matrix.

    Uses the Rodrigues formula via axis-angle conversion.

    Parameters
    ----------
    rv : RotationVector
        Rotation vector with shape (..., 3).

    Returns
    -------
    Tensor
        Rotation matrix, shape (..., 3, 3).

    Notes
    -----
    - The output matrix is orthogonal with determinant 1.
    - Uses quaternion as intermediate representation for numerical stability.

    See Also
    --------
    matrix_to_rotation_vector : Inverse conversion.
    rotation_vector_to_quaternion : Convert to quaternion.

    Examples
    --------
    Identity rotation (zero vector):

    >>> rv = rotation_vector(torch.tensor([0.0, 0.0, 0.0]))
    >>> rotation_vector_to_matrix(rv)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])

    90-degree rotation around z-axis:

    >>> import math
    >>> rv = rotation_vector(torch.tensor([0.0, 0.0, math.pi / 2]))
    >>> rotation_vector_to_matrix(rv)
    tensor([[ 0., -1.,  0.],
            [ 1.,  0.,  0.],
            [ 0.,  0.,  1.]])
    """
    # Extract angle (norm of vector) and axis (normalized vector)
    vector = rv.vector
    angle = torch.linalg.norm(vector, dim=-1)

    # Handle zero rotation case
    eps = torch.finfo(vector.dtype).eps
    angle_safe = torch.clamp(angle, min=eps)
    axis = vector / angle_safe.unsqueeze(-1)

    # For zero rotations, the axis doesn't matter (set to z-axis)
    is_zero = angle < 1e-10
    if is_zero.any():
        default_axis = torch.zeros_like(axis)
        default_axis[..., 2] = 1.0
        axis = torch.where(is_zero.unsqueeze(-1), default_axis, axis)

    # Create axis-angle and convert to matrix
    aa = AxisAngle(axis=axis, angle=angle)
    return axis_angle_to_matrix(aa)


def matrix_to_rotation_vector(matrix: Tensor) -> RotationVector:
    """Convert 3x3 rotation matrix to rotation vector.

    Uses the inverse Rodrigues formula via axis-angle conversion.

    Parameters
    ----------
    matrix : Tensor
        Rotation matrix, shape (..., 3, 3).

    Returns
    -------
    RotationVector
        Rotation vector with shape (..., 3).

    Notes
    -----
    - For valid rotation matrices, the result represents the same rotation.
    - Uses quaternion as intermediate representation for numerical stability.

    See Also
    --------
    rotation_vector_to_matrix : Inverse conversion.
    matrix_to_quaternion : Convert to quaternion.

    Examples
    --------
    Identity matrix:

    >>> R = torch.eye(3)
    >>> rv = matrix_to_rotation_vector(R)
    >>> rv.vector
    tensor([0., 0., 0.])

    Round-trip conversion:

    >>> import math
    >>> rv = rotation_vector(torch.tensor([0.0, 0.0, math.pi / 2]))
    >>> R = rotation_vector_to_matrix(rv)
    >>> rv_back = matrix_to_rotation_vector(R)
    >>> rotation_vector_to_matrix(rv_back)
    tensor([[ 0., -1.,  0.],
            [ 1.,  0.,  0.],
            [ 0.,  0.,  1.]])
    """
    aa = matrix_to_axis_angle(matrix)

    # Rotation vector = axis * angle
    angle = aa.angle.unsqueeze(-1)
    vector = aa.axis * angle

    return RotationVector(vector=vector)

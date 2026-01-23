"""SE(3) rigid body transform representation and operations.

SE(3) (Special Euclidean group in 3D) represents rigid body transformations
consisting of rotations and translations in 3D space.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.geometry.transform._quaternion import (
    Quaternion,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_slerp,
    quaternion_to_matrix,
)


@tensorclass
class RigidTransform:
    """SE(3) rigid body transform (rotation + translation).

    Represents a 3D rigid body transformation consisting of a rotation
    (represented by a unit quaternion) and a translation vector.

    The transformation applies rotation first, then translation:

    .. math::

        T(p) = R \\cdot p + t

    where :math:`R` is the rotation matrix corresponding to the quaternion
    and :math:`t` is the translation vector.

    Attributes
    ----------
    rotation : Quaternion
        Unit quaternion representing the rotation, shape (..., 4).
    translation : Tensor
        Translation vector, shape (..., 3).

    Examples
    --------
    Identity transform:
        RigidTransform(
            rotation=Quaternion(wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0])),
            translation=torch.zeros(3)
        )

    Transform with 90-degree rotation around z and translation:
        RigidTransform(
            rotation=Quaternion(wxyz=torch.tensor([0.7071, 0.0, 0.0, 0.7071])),
            translation=torch.tensor([1.0, 2.0, 3.0])
        )

    Batch of transforms:
        RigidTransform(
            rotation=Quaternion(wxyz=torch.randn(100, 4)),
            translation=torch.randn(100, 3)
        )
    """

    rotation: Quaternion
    translation: Tensor


def rigid_transform(
    rotation: Quaternion, translation: Tensor
) -> RigidTransform:
    """Create a rigid transform from rotation and translation.

    Parameters
    ----------
    rotation : Quaternion
        Unit quaternion representing the rotation, shape (..., 4).
    translation : Tensor
        Translation vector, shape (..., 3).

    Returns
    -------
    RigidTransform
        RigidTransform instance.

    Raises
    ------
    ValueError
        If translation does not have last dimension 3.
        If rotation and translation batch shapes cannot broadcast.

    Examples
    --------
    >>> from torchscience.geometry.transform import quaternion
    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> t = torch.tensor([1.0, 2.0, 3.0])
    >>> transform = rigid_transform(q, t)
    >>> transform.translation
    tensor([1., 2., 3.])
    """
    if translation.shape[-1] != 3:
        raise ValueError(
            f"rigid_transform: translation must have last dimension 3, "
            f"got {translation.shape[-1]}"
        )

    # Check that batch shapes can broadcast
    rot_batch = rotation.wxyz.shape[:-1]
    trans_batch = translation.shape[:-1]
    try:
        torch.broadcast_shapes(rot_batch, trans_batch)
    except RuntimeError:
        raise ValueError(
            f"rigid_transform: rotation batch shape {rot_batch} and "
            f"translation batch shape {trans_batch} cannot broadcast"
        )

    return RigidTransform(rotation=rotation, translation=translation)


def rigid_transform_identity(
    batch_shape: Tuple[int, ...] = (),
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> RigidTransform:
    """Create an identity rigid transform.

    The identity transform has identity rotation (no rotation) and zero
    translation. When applied to a point, it returns the point unchanged.

    Parameters
    ----------
    batch_shape : tuple of int, optional
        Batch shape for the transform. Default is () for a single transform.
    device : torch.device, optional
        Device for the tensors. Default is CPU.
    dtype : torch.dtype, optional
        Data type for the tensors. Default is float32.

    Returns
    -------
    RigidTransform
        Identity transform with the specified batch shape.

    Examples
    --------
    Single identity transform:

    >>> identity = rigid_transform_identity()
    >>> identity.rotation.wxyz
    tensor([1., 0., 0., 0.])
    >>> identity.translation
    tensor([0., 0., 0.])

    Batch of identity transforms:

    >>> batch = rigid_transform_identity(batch_shape=(10,))
    >>> batch.rotation.wxyz.shape
    torch.Size([10, 4])
    """
    # Identity quaternion: [1, 0, 0, 0]
    q_wxyz = torch.zeros((*batch_shape, 4), device=device, dtype=dtype)
    q_wxyz[..., 0] = 1.0

    # Zero translation
    t = torch.zeros((*batch_shape, 3), device=device, dtype=dtype)

    return RigidTransform(rotation=Quaternion(wxyz=q_wxyz), translation=t)


def rigid_transform_compose(
    t1: RigidTransform, t2: RigidTransform
) -> RigidTransform:
    """Compose two rigid transforms.

    The composition T1 * T2 represents applying T2 first, then T1.
    This follows the convention that T * p = T1 * (T2 * p).

    The composition is computed as:

    .. math::

        (R_1, t_1) \\cdot (R_2, t_2) = (R_1 R_2, R_1 t_2 + t_1)

    Parameters
    ----------
    t1 : RigidTransform
        First transform (applied second).
    t2 : RigidTransform
        Second transform (applied first).

    Returns
    -------
    RigidTransform
        Composed transform T1 * T2.

    Notes
    -----
    - Transform composition is **not commutative**: T1 * T2 != T2 * T1.
    - When applied to points: (T1 * T2)(p) = T1(T2(p)).

    Examples
    --------
    Compose two translations:

    >>> from torchscience.geometry.transform import quaternion
    >>> q_id = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> t1 = rigid_transform(q_id, torch.tensor([1.0, 0.0, 0.0]))
    >>> t2 = rigid_transform(q_id, torch.tensor([0.0, 1.0, 0.0]))
    >>> composed = rigid_transform_compose(t1, t2)
    >>> composed.translation
    tensor([1., 1., 0.])
    """
    # Compose rotations: R1 * R2
    rotation = quaternion_multiply(t1.rotation, t2.rotation)

    # Compose translations: R1 * t2 + t1
    rotated_t2 = quaternion_apply(t1.rotation, t2.translation)
    translation = rotated_t2 + t1.translation

    return RigidTransform(rotation=rotation, translation=translation)


def rigid_transform_inverse(t: RigidTransform) -> RigidTransform:
    """Compute the inverse of a rigid transform.

    The inverse transform T^(-1) satisfies T * T^(-1) = T^(-1) * T = Identity.

    The inverse is computed as:

    .. math::

        (R, t)^{-1} = (R^{-1}, -R^{-1} t)

    Parameters
    ----------
    t : RigidTransform
        Transform to invert.

    Returns
    -------
    RigidTransform
        Inverse transform.

    Examples
    --------
    >>> from torchscience.geometry.transform import quaternion
    >>> q = quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    >>> transform = rigid_transform(q, torch.tensor([1.0, 2.0, 3.0]))
    >>> inv = rigid_transform_inverse(transform)
    >>> composed = rigid_transform_compose(transform, inv)
    >>> torch.allclose(composed.translation, torch.zeros(3), atol=1e-5)
    True
    """
    # Inverse rotation: R^(-1) = conjugate for unit quaternion
    rotation_inv = quaternion_inverse(t.rotation)

    # Inverse translation: -R^(-1) * t
    translation_inv = -quaternion_apply(rotation_inv, t.translation)

    return RigidTransform(rotation=rotation_inv, translation=translation_inv)


def rigid_transform_apply(t: RigidTransform, points: Tensor) -> Tensor:
    """Apply a rigid transform to 3D points.

    Applies the transformation T(p) = R * p + t, where R is the rotation
    and t is the translation.

    Parameters
    ----------
    t : RigidTransform
        Rigid transform to apply.
    points : Tensor
        3D points, shape (..., 3).

    Returns
    -------
    Tensor
        Transformed points, shape (..., 3).

    Notes
    -----
    - Batch dimensions are broadcast between transform and points.
    - Rotation is applied first, then translation.

    Examples
    --------
    Apply identity transform (returns original point):

    >>> identity = rigid_transform_identity()
    >>> point = torch.tensor([1.0, 2.0, 3.0])
    >>> rigid_transform_apply(identity, point)
    tensor([1., 2., 3.])

    Apply translation:

    >>> from torchscience.geometry.transform import quaternion
    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> transform = rigid_transform(q, torch.tensor([1.0, 0.0, 0.0]))
    >>> rigid_transform_apply(transform, torch.zeros(3))
    tensor([1., 0., 0.])
    """
    # Apply rotation
    rotated = quaternion_apply(t.rotation, points)

    # Add translation
    return rotated + t.translation


def rigid_transform_apply_vector(t: RigidTransform, vectors: Tensor) -> Tensor:
    """Apply only the rotation part of a rigid transform to vectors.

    Unlike points, vectors are not affected by translation. This applies
    only the rotation: R * v.

    Parameters
    ----------
    t : RigidTransform
        Rigid transform (only rotation is used).
    vectors : Tensor
        3D vectors, shape (..., 3).

    Returns
    -------
    Tensor
        Rotated vectors, shape (..., 3).

    Notes
    -----
    - Translation is ignored for vectors.
    - Useful for transforming normals, directions, velocities, etc.
    - The vector length is preserved.

    Examples
    --------
    Apply to direction vector (translation is ignored):

    >>> from torchscience.geometry.transform import quaternion
    >>> import math
    >>> q = quaternion(torch.tensor([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)]))
    >>> t = torch.tensor([100.0, 200.0, 300.0])  # large translation
    >>> transform = rigid_transform(q, t)
    >>> vector = torch.tensor([1.0, 0.0, 0.0])
    >>> rigid_transform_apply_vector(transform, vector)
    tensor([0., 1., 0.])  # rotated, but not translated
    """
    return quaternion_apply(t.rotation, vectors)


def rigid_transform_to_matrix(t: RigidTransform) -> Tensor:
    """Convert rigid transform to 4x4 homogeneous transformation matrix.

    The matrix has the form:

    .. math::

        \\begin{bmatrix}
        R & t \\\\
        0 & 1
        \\end{bmatrix}

    where R is the 3x3 rotation matrix and t is the translation vector.

    Parameters
    ----------
    t : RigidTransform
        Rigid transform to convert.

    Returns
    -------
    Tensor
        4x4 homogeneous transformation matrix, shape (..., 4, 4).

    Examples
    --------
    Identity transform gives identity matrix:

    >>> identity = rigid_transform_identity()
    >>> rigid_transform_to_matrix(identity)
    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])
    """
    # Get rotation matrix
    R = quaternion_to_matrix(t.rotation)  # (..., 3, 3)

    # Get batch shape
    batch_shape = R.shape[:-2]

    # Create 4x4 matrix
    matrix = torch.zeros((*batch_shape, 4, 4), device=R.device, dtype=R.dtype)

    # Fill in rotation part
    matrix[..., :3, :3] = R

    # Fill in translation part
    matrix[..., :3, 3] = t.translation

    # Fill in homogeneous row
    matrix[..., 3, 3] = 1.0

    return matrix


def rigid_transform_from_matrix(matrix: Tensor) -> RigidTransform:
    """Create rigid transform from 4x4 homogeneous transformation matrix.

    Parameters
    ----------
    matrix : Tensor
        4x4 homogeneous transformation matrix, shape (..., 4, 4).

    Returns
    -------
    RigidTransform
        Rigid transform extracted from the matrix.

    Raises
    ------
    ValueError
        If matrix does not have shape (..., 4, 4).

    Notes
    -----
    - The rotation matrix must be a valid SO(3) matrix for correct results.
    - The last row should be [0, 0, 0, 1] for a valid SE(3) matrix.

    Examples
    --------
    Convert identity matrix:

    >>> matrix = torch.eye(4)
    >>> transform = rigid_transform_from_matrix(matrix)
    >>> transform.translation
    tensor([0., 0., 0.])
    """
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(
            f"rigid_transform_from_matrix: matrix must have shape (..., 4, 4), "
            f"got {matrix.shape}"
        )

    # Extract rotation matrix
    R = matrix[..., :3, :3]

    # Convert to quaternion
    rotation = matrix_to_quaternion(R)

    # Extract translation
    translation = matrix[..., :3, 3]

    return RigidTransform(rotation=rotation, translation=translation)


def rigid_transform_slerp(
    t1: RigidTransform, t2: RigidTransform, alpha: Tensor
) -> RigidTransform:
    """Interpolate between two rigid transforms.

    Uses SLERP (spherical linear interpolation) for the rotation component
    and linear interpolation for the translation component.

    Parameters
    ----------
    t1 : RigidTransform
        Start transform (alpha=0).
    t2 : RigidTransform
        End transform (alpha=1).
    alpha : Tensor
        Interpolation parameter, typically in [0, 1].

    Returns
    -------
    RigidTransform
        Interpolated transform.

    Notes
    -----
    - At alpha=0, returns t1.
    - At alpha=1, returns t2.
    - At alpha=0.5, returns the midpoint transform.
    - Rotation uses SLERP for smooth interpolation on SO(3).
    - Translation uses simple linear interpolation.

    Examples
    --------
    Interpolate between identity and a rotated/translated transform:

    >>> from torchscience.geometry.transform import quaternion
    >>> import math
    >>> q1 = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> q2 = quaternion(torch.tensor([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)]))
    >>> t1_val = rigid_transform(q1, torch.zeros(3))
    >>> t2_val = rigid_transform(q2, torch.tensor([2.0, 0.0, 0.0]))
    >>> mid = rigid_transform_slerp(t1_val, t2_val, torch.tensor(0.5))
    >>> mid.translation
    tensor([1., 0., 0.])
    """
    # SLERP for rotation
    rotation = quaternion_slerp(t1.rotation, t2.rotation, alpha)

    # Linear interpolation for translation
    # Handle broadcasting of alpha
    alpha_expanded = alpha
    while alpha_expanded.dim() < t1.translation.dim():
        alpha_expanded = alpha_expanded.unsqueeze(-1)

    translation = (
        1 - alpha_expanded
    ) * t1.translation + alpha_expanded * t2.translation

    return RigidTransform(rotation=rotation, translation=translation)

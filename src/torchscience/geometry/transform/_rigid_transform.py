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


def rigid_transform_to_dual_quaternion(t: RigidTransform) -> Tensor:
    """Convert rigid transform to dual quaternion representation.

    A dual quaternion is a compact 8-dimensional representation of SE(3) rigid
    body transformations, consisting of a real quaternion (rotation) and a dual
    quaternion (encoding translation).

    The dual quaternion is computed as:

    .. math::

        \\hat{q} = (q_r, q_d)

    where:
    - :math:`q_r` is the rotation quaternion (the real part)
    - :math:`q_d = \\frac{1}{2} q_t \\cdot q_r` where :math:`q_t = (0, t)` is the
      pure quaternion formed from the translation vector

    Parameters
    ----------
    t : RigidTransform
        Rigid transform to convert.

    Returns
    -------
    Tensor
        Dual quaternion, shape (..., 8) where:
        - [..., :4] is the real part (rotation quaternion, wxyz convention)
        - [..., 4:] is the dual part

    Notes
    -----
    - Dual quaternions provide a singularity-free representation of SE(3).
    - They are commonly used in skeletal animation, robotics, and
      smooth interpolation of rigid body motions.
    - The real part is always a unit quaternion for valid SE(3) transforms.
    - Uses wxyz convention for both the real and dual quaternion parts.

    See Also
    --------
    dual_quaternion_to_rigid_transform : Convert dual quaternion to rigid transform.
    rigid_transform_to_matrix : Convert to 4x4 matrix representation.

    References
    ----------
    .. [1] Kenwright, Ben. "A Beginners Guide to Dual-Quaternions."
           Conference on Computer Graphics, Visualization and Computer Vision, 2012.
    .. [2] https://en.wikipedia.org/wiki/Dual_quaternion

    Examples
    --------
    Identity transform gives [1,0,0,0, 0,0,0,0]:

    >>> identity = rigid_transform_identity()
    >>> rigid_transform_to_dual_quaternion(identity)
    tensor([1., 0., 0., 0., 0., 0., 0., 0.])

    Pure translation (t = [2, 4, 6]):

    >>> from torchscience.geometry.transform import quaternion
    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> transform = rigid_transform(q, torch.tensor([2.0, 4.0, 6.0]))
    >>> dq = rigid_transform_to_dual_quaternion(transform)
    >>> dq[:4]  # Real part: identity rotation
    tensor([1., 0., 0., 0.])
    >>> dq[4:]  # Dual part: (0, t/2)
    tensor([0., 1., 2., 3.])
    """
    # Get the rotation quaternion (real part)
    q_r = t.rotation.wxyz  # (..., 4)

    # Create pure quaternion from translation: q_t = (0, translation)
    # Shape: (..., 4) with w=0
    batch_shape = t.translation.shape[:-1]
    q_t = torch.zeros(
        (*batch_shape, 4),
        device=t.translation.device,
        dtype=t.translation.dtype,
    )
    q_t[..., 1:] = t.translation  # (0, tx, ty, tz)

    # Compute dual part: q_d = (1/2) * q_t * q_r
    # Using quaternion multiplication
    q_t_quat = Quaternion(wxyz=q_t)
    q_r_quat = t.rotation
    q_d = quaternion_multiply(q_t_quat, q_r_quat).wxyz * 0.5

    # Concatenate real and dual parts
    return torch.cat([q_r, q_d], dim=-1)


def _skew_symmetric_3x3(v: Tensor) -> Tensor:
    """Compute the skew-symmetric matrix [v]_x from a 3-vector.

    Parameters
    ----------
    v : Tensor
        3-vector, shape (..., 3).

    Returns
    -------
    Tensor
        Skew-symmetric matrix, shape (..., 3, 3).
    """
    v_x = v[..., 0]
    v_y = v[..., 1]
    v_z = v[..., 2]

    zeros = torch.zeros_like(v_x)

    row0 = torch.stack([zeros, -v_z, v_y], dim=-1)
    row1 = torch.stack([v_z, zeros, -v_x], dim=-1)
    row2 = torch.stack([-v_y, v_x, zeros], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def rigid_transform_adjoint(t: RigidTransform) -> Tensor:
    """Compute the 6x6 adjoint matrix for a rigid transform.

    The adjoint representation transforms twists between reference frames.
    If xi is a twist in frame A, and T transforms from A to B, then
    Ad_T @ xi gives the twist in frame B.

    The adjoint matrix has the structure:

    .. math::

        Ad_T = \\begin{bmatrix}
            R & 0 \\\\
            [t]_\\times R & R
        \\end{bmatrix}

    where R is the 3x3 rotation matrix, [t]_x is the skew-symmetric matrix
    of the translation vector, and 0 is a 3x3 zero matrix.

    Parameters
    ----------
    t : RigidTransform
        Rigid transform (rotation and translation).

    Returns
    -------
    Tensor
        6x6 adjoint matrix, shape (..., 6, 6).

    Notes
    -----
    - The adjoint satisfies: Ad_{T1 T2} = Ad_T1 @ Ad_T2
    - The inverse relation: Ad_{T^{-1}} = (Ad_T)^{-1}
    - Used for changing reference frames for twists and wrenches
    - Fundamental in robotics for Jacobian computation and velocity propagation

    Examples
    --------
    Identity transform gives identity adjoint:

    >>> identity = rigid_transform_identity()
    >>> adjoint = rigid_transform_adjoint(identity)
    >>> torch.allclose(adjoint, torch.eye(6), atol=1e-5)
    True

    Pure rotation (no translation):

    >>> from torchscience.geometry.transform import quaternion
    >>> import math
    >>> q = quaternion(torch.tensor([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)]))
    >>> transform = rigid_transform(q, torch.zeros(3))
    >>> adjoint = rigid_transform_adjoint(transform)
    >>> # [t]_x R = 0 for zero translation
    >>> torch.allclose(adjoint[3:, :3], torch.zeros(3, 3), atol=1e-5)
    True
    """
    # Get rotation matrix from quaternion
    R = quaternion_to_matrix(t.rotation)  # (..., 3, 3)

    # Get batch shape
    batch_shape = R.shape[:-2]

    # Create 6x6 adjoint matrix
    adjoint = torch.zeros((*batch_shape, 6, 6), device=R.device, dtype=R.dtype)

    # Top-left block: R
    adjoint[..., :3, :3] = R

    # Top-right block: 0 (already initialized to zero)

    # Bottom-right block: R
    adjoint[..., 3:, 3:] = R

    # Bottom-left block: [t]_x @ R
    t_skew = _skew_symmetric_3x3(t.translation)  # (..., 3, 3)
    adjoint[..., 3:, :3] = torch.matmul(t_skew, R)

    return adjoint


def dual_quaternion_to_rigid_transform(dq: Tensor) -> RigidTransform:
    """Convert dual quaternion to rigid transform.

    Extracts the rotation and translation from an 8-dimensional dual quaternion
    representation of an SE(3) transform.

    The conversion is:

    .. math::

        q_r = \\hat{q}_{real}

        t = 2 \\cdot q_d \\cdot q_r^*

    where :math:`q_r^*` is the conjugate of the rotation quaternion.

    Parameters
    ----------
    dq : Tensor
        Dual quaternion, shape (..., 8) where:
        - [..., :4] is the real part (rotation quaternion, wxyz convention)
        - [..., 4:] is the dual part

    Returns
    -------
    RigidTransform
        Rigid transform with extracted rotation and translation.

    Raises
    ------
    ValueError
        If dq does not have last dimension 8.

    Notes
    -----
    - The real part is expected to be a unit quaternion for valid SE(3) transforms.
    - Uses wxyz convention for both the real and dual quaternion parts.
    - The extracted translation is the imaginary part of 2 * q_d * conj(q_r).

    See Also
    --------
    rigid_transform_to_dual_quaternion : Convert rigid transform to dual quaternion.
    rigid_transform_from_matrix : Convert from 4x4 matrix representation.

    References
    ----------
    .. [1] Kenwright, Ben. "A Beginners Guide to Dual-Quaternions."
           Conference on Computer Graphics, Visualization and Computer Vision, 2012.
    .. [2] https://en.wikipedia.org/wiki/Dual_quaternion

    Examples
    --------
    Identity dual quaternion:

    >>> dq = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> transform = dual_quaternion_to_rigid_transform(dq)
    >>> transform.rotation.wxyz
    tensor([1., 0., 0., 0.])
    >>> transform.translation
    tensor([0., 0., 0.])

    Dual quaternion with translation:

    >>> dq = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    >>> transform = dual_quaternion_to_rigid_transform(dq)
    >>> transform.translation
    tensor([2., 4., 6.])
    """
    if dq.shape[-1] != 8:
        raise ValueError(
            f"dual_quaternion_to_rigid_transform: dq must have last dimension 8, "
            f"got {dq.shape[-1]}"
        )

    # Extract real and dual parts
    q_r = dq[..., :4]  # Rotation quaternion (real part)
    q_d = dq[..., 4:]  # Dual part

    # Create Quaternion objects
    rotation = Quaternion(wxyz=q_r)

    # Compute translation: t = 2 * q_d * conj(q_r)
    # conj(q_r) for unit quaternion is just the inverse
    q_r_inv = quaternion_inverse(rotation)

    # Multiply: 2 * q_d * q_r_inv
    q_d_quat = Quaternion(wxyz=q_d)
    t_quat = quaternion_multiply(q_d_quat, q_r_inv)

    # Extract translation from imaginary part (xyz) and scale by 2
    translation = t_quat.wxyz[..., 1:] * 2.0

    return RigidTransform(rotation=rotation, translation=translation)

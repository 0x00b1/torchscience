"""Euler angles rotation representation and conversions.

Euler angles represent a 3D rotation as a sequence of three rotations around
coordinate axes. The convention string (e.g., "XYZ") specifies the order of
rotation axes.

There are 12 possible conventions:
- 6 Tait-Bryan angles (three different axes): XYZ, XZY, YXZ, YZX, ZXY, ZYX
- 6 Proper Euler angles (first and third axes same): XYX, XZX, YXY, YZY, ZXZ, ZYZ

The rotations are intrinsic (rotating frame), meaning each rotation is around
the axis of the rotated coordinate system.
"""

from __future__ import annotations

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.geometry.transform._conventions import (
    get_axis_indices,
    validate_convention,
)
from torchscience.geometry.transform._quaternion import (
    Quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)


@tensorclass
class EulerAngles:
    """Euler angles rotation representation.

    Represents a 3D rotation as three angles around coordinate axes.

    Attributes
    ----------
    angles : Tensor
        Euler angles in radians, shape (..., 3).
        The three angles correspond to rotations around the axes specified
        by the convention in order.
    convention : str
        Rotation order like "XYZ", "ZYX", etc.

    Examples
    --------
    XYZ Euler angles (roll, pitch, yaw):
        EulerAngles(angles=torch.tensor([0.1, 0.2, 0.3]), convention="XYZ")

    ZYX convention (common in aerospace):
        EulerAngles(angles=torch.tensor([0.1, 0.2, 0.3]), convention="ZYX")

    Batch of Euler angles:
        EulerAngles(angles=torch.randn(100, 3), convention="XYZ")

    Notes
    -----
    - The rotations are intrinsic, meaning each rotation is applied in the
      rotated coordinate frame.
    - Euler angles suffer from gimbal lock when the middle angle approaches
      +/- pi/2 (for Tait-Bryan conventions) or 0/pi (for proper Euler).
    """

    angles: Tensor
    convention: str


def euler_angles(angles: Tensor, convention: str = "XYZ") -> EulerAngles:
    """Create Euler angles from tensor.

    Parameters
    ----------
    angles : Tensor
        Euler angles in radians, shape (..., 3).
    convention : str, optional
        Rotation order (default: "XYZ"). Valid options are:
        - Tait-Bryan: "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"
        - Proper Euler: "XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ"

    Returns
    -------
    EulerAngles
        EulerAngles instance.

    Raises
    ------
    ValueError
        If angles does not have last dimension 3.
    ValueError
        If convention is not valid.

    Examples
    --------
    >>> ea = euler_angles(torch.tensor([0.1, 0.2, 0.3]), convention="XYZ")
    >>> ea.angles
    tensor([0.1000, 0.2000, 0.3000])
    >>> ea.convention
    'XYZ'
    """
    if angles.shape[-1] != 3:
        raise ValueError(
            f"euler_angles: angles must have last dimension 3, got {angles.shape[-1]}"
        )
    validate_convention(convention)
    return EulerAngles(angles=angles, convention=convention)


def _axis_angle_to_quaternion_single(
    axis_idx: int, angle: Tensor
) -> Quaternion:
    """Create quaternion from single-axis rotation.

    Parameters
    ----------
    axis_idx : int
        Axis index (0=X, 1=Y, 2=Z).
    angle : Tensor
        Rotation angle in radians, shape (...).

    Returns
    -------
    Quaternion
        Quaternion representing rotation around the specified axis.
    """
    half_angle = angle / 2.0
    c = torch.cos(half_angle)
    s = torch.sin(half_angle)

    # Build wxyz tensor
    # Shape: (..., 4) where ... is the shape of angle
    batch_shape = angle.shape
    wxyz = torch.zeros(
        (*batch_shape, 4), dtype=angle.dtype, device=angle.device
    )
    wxyz[..., 0] = c
    wxyz[..., 1 + axis_idx] = s

    return Quaternion(wxyz=wxyz)


def euler_angles_to_quaternion(ea: EulerAngles) -> Quaternion:
    """Convert Euler angles to quaternion.

    Composes three single-axis rotations according to the convention.

    Parameters
    ----------
    ea : EulerAngles
        Euler angles with shape (..., 3).

    Returns
    -------
    Quaternion
        Unit quaternion in wxyz convention, shape (..., 4).

    Notes
    -----
    - The output quaternion is always a unit quaternion.
    - For intrinsic rotations (convention "ABC"), the composition is:
      q = q_C * q_B * q_A (reverse order, as each subsequent rotation
      is in the local/body frame).

    See Also
    --------
    quaternion_to_euler_angles : Inverse conversion.
    euler_angles_to_matrix : Convert to rotation matrix.

    Examples
    --------
    Identity rotation (all zeros):

    >>> ea = euler_angles(torch.tensor([0.0, 0.0, 0.0]), convention="XYZ")
    >>> euler_angles_to_quaternion(ea).wxyz
    tensor([1., 0., 0., 0.])

    90-degree rotation around z-axis:

    >>> import math
    >>> ea = euler_angles(torch.tensor([0.0, 0.0, math.pi / 2]), convention="XYZ")
    >>> q = euler_angles_to_quaternion(ea)
    >>> q.wxyz
    tensor([0.7071, 0.0000, 0.0000, 0.7071])
    """
    i, j, k = get_axis_indices(ea.convention)

    # Extract individual angles
    angle1 = ea.angles[..., 0]
    angle2 = ea.angles[..., 1]
    angle3 = ea.angles[..., 2]

    # Create quaternion for each single-axis rotation
    q1 = _axis_angle_to_quaternion_single(i, angle1)
    q2 = _axis_angle_to_quaternion_single(j, angle2)
    q3 = _axis_angle_to_quaternion_single(k, angle3)

    # Compose: q = q3 * q2 * q1 (intrinsic rotations)
    # For intrinsic rotation sequence "ABC":
    # - First rotate around A in body frame
    # - Then rotate around B in body frame
    # - Then rotate around C in body frame
    # In global frame, this is equivalent to: C * B * A
    q = quaternion_multiply(quaternion_multiply(q3, q2), q1)

    return q


def euler_angles_to_matrix(ea: EulerAngles) -> Tensor:
    """Convert Euler angles to 3x3 rotation matrix.

    Converts via quaternion for numerical stability and consistency.

    Parameters
    ----------
    ea : EulerAngles
        Euler angles with shape (..., 3).

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
    matrix_to_euler_angles : Inverse conversion.
    euler_angles_to_quaternion : Convert to quaternion.

    Examples
    --------
    Identity rotation (all zeros):

    >>> ea = euler_angles(torch.tensor([0.0, 0.0, 0.0]), convention="XYZ")
    >>> euler_angles_to_matrix(ea)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])

    90-degree rotation around z-axis:

    >>> import math
    >>> ea = euler_angles(torch.tensor([0.0, 0.0, math.pi / 2]), convention="XYZ")
    >>> euler_angles_to_matrix(ea)
    tensor([[ 0., -1.,  0.],
            [ 1.,  0.,  0.],
            [ 0.,  0.,  1.]])
    """
    q = euler_angles_to_quaternion(ea)
    return quaternion_to_matrix(q)


def _matrix_to_euler_angles_tait_bryan(
    matrix: Tensor, i: int, j: int, k: int
) -> Tensor:
    """Extract Euler angles from rotation matrix for Tait-Bryan conventions.

    For intrinsic Tait-Bryan angles with convention "ABC" (three different axes),
    the rotation matrix is R = R_C @ R_B @ R_A.

    The extraction depends on the specific convention. We classify by:
    - Even permutation (cyclic): XYZ, YZX, ZXY  - where (j-i) % 3 == 1
    - Odd permutation: XZY, YXZ, ZYX - where (j-i) % 3 == 2
    """
    # Determine if this is an even or odd permutation
    # Even (cyclic): XYZ (012), YZX (120), ZXY (201) - (j-i) % 3 == 1
    # Odd (anti-cyclic): XZY (021), YXZ (102), ZYX (210) - (j-i) % 3 == 2
    parity = (j - i) % 3
    is_even = parity == 1

    if is_even:
        # Even permutation (XYZ, YZX, ZXY)
        # Matrix structure for intrinsic rotation (R = R_k @ R_j @ R_i):
        #   R[k,i] = -sin(b), so b = asin(-R[k,i])
        #   R[k,j] = sin(a)*cos(b), R[k,k] = cos(a)*cos(b), so a = atan2(R[k,j], R[k,k])
        #   R[j,i] = sin(c)*cos(b), R[i,i] = cos(c)*cos(b), so c = atan2(R[j,i], R[i,i])
        angle2 = torch.asin(torch.clamp(-matrix[..., k, i], -1.0, 1.0))
        angle1 = torch.atan2(matrix[..., k, j], matrix[..., k, k])
        angle3 = torch.atan2(matrix[..., j, i], matrix[..., i, i])
    else:
        # Odd permutation (XZY, YXZ, ZYX)
        # Matrix structure for intrinsic rotation (R = R_k @ R_j @ R_i):
        #   R[k,i] = sin(b), so b = asin(R[k,i])
        #   R[k,j] = -sin(a)*cos(b), R[k,k] = cos(a)*cos(b), so a = atan2(-R[k,j], R[k,k])
        #   R[j,i] = -sin(c)*cos(b), R[i,i] = cos(c)*cos(b), so c = atan2(-R[j,i], R[i,i])
        angle2 = torch.asin(torch.clamp(matrix[..., k, i], -1.0, 1.0))
        angle1 = torch.atan2(-matrix[..., k, j], matrix[..., k, k])
        angle3 = torch.atan2(-matrix[..., j, i], matrix[..., i, i])

    return torch.stack([angle1, angle2, angle3], dim=-1)


def _matrix_to_euler_angles_proper(
    matrix: Tensor, i: int, j: int, k: int
) -> Tensor:
    """Extract Euler angles from rotation matrix for proper Euler conventions.

    For intrinsic proper Euler angles with convention "ABA" (first and third axes same),
    the rotation matrix is R = R_A(c) @ R_B(b) @ R_A(a) where A != B.

    Note: i == k for proper Euler conventions.

    The extraction formulas depend on the specific convention:
    - Even (cyclic): XYX, YZY, ZXZ - where (j-i) % 3 == 1
    - Odd (anti-cyclic): XZX, YXY, ZYZ - where (j-i) % 3 == 2
    """
    # For proper Euler, i == k
    # other is the third axis (not i or j)
    other = 3 - i - j

    # R[i,i] = cos(b) for all proper Euler conventions
    angle2 = torch.acos(torch.clamp(matrix[..., i, i], -1.0, 1.0))

    # Determine parity: (j-i) % 3 == 1 is even/cyclic, 2 is odd/anti-cyclic
    parity = (j - i) % 3
    is_even = parity == 1

    if is_even:
        # Even/cyclic: XYX (j=1 > i=0), YZY (j=2 > i=1), ZXZ (j=0, i=2 wraps)
        # Based on matrix analysis:
        # R[i,j] = sin(a)*sin(b)
        # R[i,other] = cos(a)*sin(b)  -> a = atan2(R[i,j], R[i,other])
        # R[j,i] = sin(c)*sin(b)
        # R[other,i] = -cos(c)*sin(b) -> c = atan2(R[j,i], -R[other,i])
        angle1 = torch.atan2(matrix[..., i, j], matrix[..., i, other])
        angle3 = torch.atan2(matrix[..., j, i], -matrix[..., other, i])
    else:
        # Odd/anti-cyclic: XZX (j=2, i=0), YXY (j=0, i=1), ZYZ (j=1, i=2)
        # Based on matrix analysis:
        # R[i,j] = sin(a)*sin(b)
        # R[i,other] = -cos(a)*sin(b) -> a = atan2(R[i,j], -R[i,other])
        # R[j,i] = sin(c)*sin(b)
        # R[other,i] = cos(c)*sin(b)  -> c = atan2(R[j,i], R[other,i])
        angle1 = torch.atan2(matrix[..., i, j], -matrix[..., i, other])
        angle3 = torch.atan2(matrix[..., j, i], matrix[..., other, i])

    return torch.stack([angle1, angle2, angle3], dim=-1)


def quaternion_to_euler_angles(q: Quaternion, convention: str) -> EulerAngles:
    """Convert quaternion to Euler angles.

    Extracts Euler angles from the quaternion via rotation matrix.

    Parameters
    ----------
    q : Quaternion
        Unit quaternion in wxyz convention, shape (..., 4).
    convention : str
        Euler angle convention (e.g., "XYZ", "ZYX").

    Returns
    -------
    EulerAngles
        Euler angles with shape (..., 3).

    Notes
    -----
    - Euler angle extraction can be ambiguous near gimbal lock singularities.
    - For Tait-Bryan angles, gimbal lock occurs when the middle angle is
      near +/- pi/2.
    - For proper Euler angles, gimbal lock occurs when the middle angle is
      near 0 or pi.

    See Also
    --------
    euler_angles_to_quaternion : Inverse conversion.
    matrix_to_euler_angles : Convert from rotation matrix.

    Examples
    --------
    Identity quaternion:

    >>> from torchscience.geometry.transform import quaternion
    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> ea = quaternion_to_euler_angles(q, convention="XYZ")
    >>> ea.angles
    tensor([0., 0., 0.])
    """
    validate_convention(convention)
    matrix = quaternion_to_matrix(q)
    return matrix_to_euler_angles(matrix, convention)


def matrix_to_euler_angles(matrix: Tensor, convention: str) -> EulerAngles:
    """Convert 3x3 rotation matrix to Euler angles.

    Parameters
    ----------
    matrix : Tensor
        Rotation matrix, shape (..., 3, 3).
    convention : str
        Euler angle convention (e.g., "XYZ", "ZYX").

    Returns
    -------
    EulerAngles
        Euler angles with shape (..., 3).

    Notes
    -----
    - Euler angle extraction can be ambiguous near gimbal lock singularities.
    - The returned angles may not exactly match the original angles used to
      create the matrix due to gimbal lock or angle wrapping, but they will
      produce the same rotation.

    See Also
    --------
    euler_angles_to_matrix : Inverse conversion.
    quaternion_to_euler_angles : Convert from quaternion.

    Examples
    --------
    Identity matrix:

    >>> R = torch.eye(3)
    >>> ea = matrix_to_euler_angles(R, convention="XYZ")
    >>> ea.angles
    tensor([0., 0., 0.])

    Round-trip conversion:

    >>> ea = euler_angles(torch.tensor([0.3, 0.4, 0.5]), convention="XYZ")
    >>> R = euler_angles_to_matrix(ea)
    >>> ea_back = matrix_to_euler_angles(R, convention="XYZ")
    >>> torch.allclose(euler_angles_to_matrix(ea_back), R)
    True
    """
    validate_convention(convention)
    i, j, k = get_axis_indices(convention)

    # Check if it's Tait-Bryan (three different axes) or proper Euler (i == k)
    if i == k:
        # Proper Euler
        angles = _matrix_to_euler_angles_proper(matrix, i, j, k)
    else:
        # Tait-Bryan
        angles = _matrix_to_euler_angles_tait_bryan(matrix, i, j, k)

    return EulerAngles(angles=angles, convention=convention)

"""Quaternion representation and operations."""

from __future__ import annotations

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


@tensorclass
class Quaternion:
    """Unit quaternion representing a 3D rotation.

    Uses scalar-first (wxyz) convention: q = w + xi + yj + zk.

    Attributes
    ----------
    wxyz : Tensor
        Quaternion components in [w, x, y, z] order, shape (..., 4).
        For unit quaternions: w^2 + x^2 + y^2 + z^2 = 1.

    Examples
    --------
    Identity rotation:
        Quaternion(wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]))

    90-degree rotation around z-axis:
        Quaternion(wxyz=torch.tensor([0.7071, 0.0, 0.0, 0.7071]))

    Batch of quaternions:
        Quaternion(wxyz=torch.randn(100, 4))
    """

    wxyz: Tensor


def quaternion(wxyz: Tensor) -> Quaternion:
    """Create quaternion from wxyz tensor.

    Parameters
    ----------
    wxyz : Tensor
        Quaternion components [w, x, y, z], shape (..., 4).

    Returns
    -------
    Quaternion
        Quaternion instance.

    Raises
    ------
    ValueError
        If wxyz does not have last dimension 4.

    Examples
    --------
    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> q.wxyz
    tensor([1., 0., 0., 0.])
    """
    if wxyz.shape[-1] != 4:
        raise ValueError(
            f"quaternion: wxyz must have last dimension 4, got {wxyz.shape[-1]}"
        )
    return Quaternion(wxyz=wxyz)


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Multiply two quaternions using the Hamilton product.

    .. math::

        (q_1 \\cdot q_2)_w = w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2

        (q_1 \\cdot q_2)_x = w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2

        (q_1 \\cdot q_2)_y = w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2

        (q_1 \\cdot q_2)_z = w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2

    Where :math:`q_1 = w_1 + x_1 i + y_1 j + z_1 k` and
    :math:`q_2 = w_2 + x_2 i + y_2 j + z_2 k`.

    Parameters
    ----------
    q1 : Quaternion
        First quaternion, shape (..., 4).
    q2 : Quaternion
        Second quaternion, shape (..., 4). Batch dimensions broadcast with q1.

    Returns
    -------
    Quaternion
        Product q1 * q2, shape is broadcast of q1 and q2 batch dimensions.

    Notes
    -----
    - Quaternion multiplication is **non-commutative**: q1 * q2 != q2 * q1.
    - For unit quaternions representing rotations, q1 * q2 represents
      rotation q1 followed by rotation q2.
    - Works for any quaternions, not just unit quaternions.

    See Also
    --------
    quaternion_inverse : Compute quaternion inverse.
    quaternion_normalize : Normalize to unit quaternion.

    References
    ----------
    .. [1] Hamilton, W.R. "Elements of Quaternions", 1866.
    .. [2] https://en.wikipedia.org/wiki/Quaternion#Hamilton_product

    Examples
    --------
    >>> q1 = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # identity
    >>> q2 = quaternion(torch.tensor([0.7071, 0.7071, 0.0, 0.0]))  # 90deg around x
    >>> quaternion_multiply(q1, q2).wxyz
    tensor([0.7071, 0.7071, 0.0000, 0.0000])
    """
    result = torch.ops.torchscience.quaternion_multiply(q1.wxyz, q2.wxyz)
    return Quaternion(wxyz=result)


def quaternion_inverse(q: Quaternion) -> Quaternion:
    """Compute inverse of a unit quaternion.

    For unit quaternions, the inverse equals the conjugate:

    .. math::

        q^{-1} = [w, -x, -y, -z]

    Where :math:`q = w + xi + yj + zk` is a unit quaternion with
    :math:`|q| = w^2 + x^2 + y^2 + z^2 = 1`.

    Parameters
    ----------
    q : Quaternion
        Unit quaternion, shape (..., 4).

    Returns
    -------
    Quaternion
        Inverse quaternion q^(-1), shape (..., 4).

    Notes
    -----
    - Only valid for unit quaternions (|q| = 1).
    - For general quaternions, use q* / |q|^2.
    - The inverse satisfies q * q^(-1) = q^(-1) * q = identity.

    See Also
    --------
    quaternion_multiply : Multiply two quaternions.
    quaternion_normalize : Normalize to unit quaternion.

    References
    ----------
    .. [1] Hamilton, W.R. "Elements of Quaternions", 1866.
    .. [2] https://en.wikipedia.org/wiki/Quaternion#Conjugation,_the_norm,_and_reciprocal

    Examples
    --------
    >>> q = quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    >>> q_inv = quaternion_inverse(q)
    >>> q_inv.wxyz
    tensor([ 0.5000, -0.5000, -0.5000, -0.5000])

    Verify that q * q^(-1) gives identity:

    >>> quaternion_multiply(q, q_inv).wxyz
    tensor([1., 0., 0., 0.])
    """
    result = torch.ops.torchscience.quaternion_inverse(q.wxyz)
    return Quaternion(wxyz=result)


def quaternion_normalize(q: Quaternion) -> Quaternion:
    """Normalize a quaternion to unit length.

    Computes the unit quaternion by dividing by its norm:

    .. math::

        \\hat{q} = \\frac{q}{\\|q\\|}

    Where :math:`\\|q\\| = \\sqrt{w^2 + x^2 + y^2 + z^2}` is the quaternion norm.

    Parameters
    ----------
    q : Quaternion
        Input quaternion, shape (..., 4).

    Returns
    -------
    Quaternion
        Unit quaternion with ||q|| = 1, shape (..., 4).

    Notes
    -----
    - The output quaternion has unit norm: ||output|| = 1.
    - Normalizing an already-normalized quaternion returns the same quaternion
      (up to numerical precision).
    - For zero quaternions, the result is undefined (division by zero).
    - Useful for ensuring quaternions remain valid rotations after
      numerical operations that may cause drift from unit norm.

    See Also
    --------
    quaternion_inverse : Compute quaternion inverse.
    quaternion_multiply : Multiply two quaternions.

    References
    ----------
    .. [1] Hamilton, W.R. "Elements of Quaternions", 1866.
    .. [2] https://en.wikipedia.org/wiki/Quaternion#Unit_quaternion

    Examples
    --------
    >>> q = quaternion(torch.tensor([2.0, 0.0, 0.0, 0.0]))
    >>> q_norm = quaternion_normalize(q)
    >>> q_norm.wxyz
    tensor([1., 0., 0., 0.])

    Normalize a general quaternion:

    >>> q = quaternion(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    >>> q_norm = quaternion_normalize(q)
    >>> torch.linalg.norm(q_norm.wxyz)
    tensor(1.)
    """
    result = torch.ops.torchscience.quaternion_normalize(q.wxyz)
    return Quaternion(wxyz=result)


def quaternion_to_matrix(q: Quaternion) -> Tensor:
    """Convert quaternion to 3x3 rotation matrix.

    Converts a unit quaternion to its equivalent 3x3 rotation matrix.

    The rotation matrix R is computed as:

    .. math::

        R = \\begin{bmatrix}
        1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\\\
        2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\\\
        2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
        \\end{bmatrix}

    where :math:`q = [w, x, y, z]` is the quaternion in scalar-first convention.

    Parameters
    ----------
    q : Quaternion
        Unit quaternion(s), shape (..., 4).

    Returns
    -------
    Tensor
        Rotation matrix, shape (..., 3, 3).

    Notes
    -----
    - For unit quaternions, the resulting matrix is orthogonal: R @ R.T = I.
    - The determinant of the rotation matrix is 1.
    - Quaternions q and -q produce the same rotation matrix.

    See Also
    --------
    quaternion_apply : Apply quaternion rotation to points.
    quaternion_multiply : Multiply two quaternions.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix

    Examples
    --------
    Identity quaternion gives identity matrix:

    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> quaternion_to_matrix(q)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])

    180-degree rotation around z-axis:

    >>> q = quaternion(torch.tensor([0.0, 0.0, 0.0, 1.0]))
    >>> quaternion_to_matrix(q)
    tensor([[-1.,  0.,  0.],
            [ 0., -1.,  0.],
            [ 0.,  0.,  1.]])

    Batch of quaternions:

    >>> q = quaternion(torch.randn(10, 4))
    >>> q = quaternion_normalize(q)
    >>> R = quaternion_to_matrix(q)
    >>> R.shape
    torch.Size([10, 3, 3])
    """
    return torch.ops.torchscience.quaternion_to_matrix(q.wxyz)


def quaternion_apply(q: Quaternion, point: Tensor) -> Tensor:
    """Apply quaternion rotation to 3D points.

    Rotates point(s) by the rotation represented by quaternion(s).

    .. math::

        v' = q \\cdot [0, v] \\cdot q^{-1}

    This is computed using the optimized formula:

    .. math::

        v' = v + 2w(q_{xyz} \\times v) + 2(q_{xyz} \\times (q_{xyz} \\times v))

    where :math:`q = [w, x, y, z]` and :math:`q_{xyz} = [x, y, z]`.

    Parameters
    ----------
    q : Quaternion
        Unit quaternion(s), shape (..., 4).
    point : Tensor
        3D point(s), shape (..., 3). Batch dimensions broadcast with q.

    Returns
    -------
    Tensor
        Rotated point(s), shape is broadcast of q and point batch dims, plus 3.

    Notes
    -----
    - Requires unit quaternions (|q| = 1) for correct rotation.
    - Uses an optimized formula that avoids quaternion multiplication.
    - Batch dimensions are broadcast between q and point.

    See Also
    --------
    quaternion_multiply : Multiply two quaternions.
    quaternion_inverse : Compute quaternion inverse.
    quaternion_normalize : Normalize to unit quaternion.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Examples
    --------
    Identity rotation (returns original point):

    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> point = torch.tensor([1.0, 0.0, 0.0])
    >>> quaternion_apply(q, point)
    tensor([1., 0., 0.])

    90-degree rotation around x-axis (maps y to z):

    >>> import math
    >>> q = quaternion(torch.tensor([math.cos(math.pi/4), math.sin(math.pi/4), 0.0, 0.0]))
    >>> point = torch.tensor([0.0, 1.0, 0.0])
    >>> quaternion_apply(q, point)
    tensor([0., 0., 1.])
    """
    return torch.ops.torchscience.quaternion_apply(q.wxyz, point)


def matrix_to_quaternion(matrix: Tensor) -> Quaternion:
    """Convert 3x3 rotation matrix to quaternion.

    Uses Shepperd's method for numerical stability by choosing the largest
    diagonal element to avoid division by small numbers.

    Parameters
    ----------
    matrix : Tensor
        Rotation matrix, shape (..., 3, 3).

    Returns
    -------
    Quaternion
        Unit quaternion, shape (..., 4) in wxyz convention.

    Notes
    -----
    - For valid rotation matrices (orthogonal with determinant 1), the output
      is a unit quaternion.
    - Rotation matrices R and the output quaternion q satisfy the property
      that quaternion_to_matrix(matrix_to_quaternion(R)) == R (up to numerical
      precision).
    - The sign of the quaternion is not uniquely determined: q and -q represent
      the same rotation.

    See Also
    --------
    quaternion_to_matrix : Convert quaternion to rotation matrix.
    quaternion_normalize : Normalize to unit quaternion.

    References
    ----------
    .. [1] Shepperd, S.W. "Quaternion from Rotation Matrix." Journal of
           Guidance and Control, 1978.
    .. [2] https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    Identity matrix gives identity quaternion (or its negative):

    >>> R = torch.eye(3)
    >>> q = matrix_to_quaternion(R)
    >>> q.wxyz
    tensor([1., 0., 0., 0.])

    Round-trip conversion:

    >>> q_orig = quaternion_normalize(quaternion(torch.randn(4)))
    >>> R = quaternion_to_matrix(q_orig)
    >>> q_back = matrix_to_quaternion(R)
    >>> # q_back should match q_orig or -q_orig
    >>> torch.allclose(q_back.wxyz, q_orig.wxyz, atol=1e-5) or \\
    ...     torch.allclose(q_back.wxyz, -q_orig.wxyz, atol=1e-5)
    True

    Batch of rotation matrices:

    >>> R = torch.randn(10, 3, 3)
    >>> q = matrix_to_quaternion(R)
    >>> q.wxyz.shape
    torch.Size([10, 4])
    """
    result = torch.ops.torchscience.matrix_to_quaternion(matrix)
    return Quaternion(wxyz=result)


def quaternion_slerp(q1: Quaternion, q2: Quaternion, t: Tensor) -> Quaternion:
    """Spherical linear interpolation between two quaternions.

    Smoothly interpolates from q1 (t=0) to q2 (t=1) along the shortest arc
    on the 4D unit sphere.

    .. math::

        \\text{slerp}(q_1, q_2, t) = \\frac{\\sin((1-t)\\theta)}{\\sin(\\theta)} q_1
        + \\frac{\\sin(t\\theta)}{\\sin(\\theta)} q_2

    Where :math:`\\theta = \\arccos(q_1 \\cdot q_2)` is the angle between the
    two quaternions.

    Parameters
    ----------
    q1 : Quaternion
        Start quaternion, shape (..., 4).
    q2 : Quaternion
        End quaternion, shape (..., 4). Batch dimensions broadcast with q1.
    t : Tensor
        Interpolation parameter (0 to 1), scalar or broadcastable shape.

    Returns
    -------
    Quaternion
        Interpolated quaternion.

    Notes
    -----
    - At t=0, returns q1. At t=1, returns q2.
    - At t=0.5, returns the midpoint rotation on the geodesic.
    - Always takes the shorter path on the sphere (negates q2 if needed).
    - Falls back to linear interpolation when quaternions are nearly parallel.
    - Output is normalized to ensure unit quaternion.

    See Also
    --------
    quaternion_multiply : Multiply two quaternions.
    quaternion_normalize : Normalize to unit quaternion.

    References
    ----------
    .. [1] Shoemake, Ken. "Animating rotation with quaternion curves."
           ACM SIGGRAPH Computer Graphics 19.3 (1985): 245-254.
    .. [2] https://en.wikipedia.org/wiki/Slerp

    Examples
    --------
    Interpolate between identity and 90-degree rotation around z-axis:

    >>> q1 = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # identity
    >>> q2 = quaternion(torch.tensor([0.7071, 0.0, 0.0, 0.7071]))  # 90 deg z
    >>> t = torch.tensor(0.0)
    >>> quaternion_slerp(q1, q2, t).wxyz
    tensor([1., 0., 0., 0.])

    >>> t = torch.tensor(1.0)
    >>> quaternion_slerp(q1, q2, t).wxyz
    tensor([0.7071, 0.0000, 0.0000, 0.7071])

    Batch interpolation:

    >>> q1 = quaternion(torch.randn(10, 4))
    >>> q1 = quaternion_normalize(q1)
    >>> q2 = quaternion(torch.randn(10, 4))
    >>> q2 = quaternion_normalize(q2)
    >>> t = torch.linspace(0, 1, 10)
    >>> result = quaternion_slerp(q1, q2, t)
    >>> result.wxyz.shape
    torch.Size([10, 4])
    """
    result = torch.ops.torchscience.quaternion_slerp(q1.wxyz, q2.wxyz, t)
    return Quaternion(wxyz=result)

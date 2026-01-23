"""6D continuous rotation representation and conversions.

Reference: Zhou et al., "On the Continuity of Rotation Representations in
Neural Networks", CVPR 2019.
"""

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
class Rotation6D:
    """6D continuous rotation representation.

    Represents a 3D rotation using the first two columns of its rotation
    matrix, flattened into a 6-dimensional vector. This representation is
    continuous (no singularities), making it well-suited for neural network
    training.

    The 6D representation is defined as:

    .. math::

        \\mathbf{r} = [\\mathbf{r}_1, \\mathbf{r}_2] \\in \\mathbb{R}^6

    where :math:`\\mathbf{r}_1, \\mathbf{r}_2 \\in \\mathbb{R}^3` are the first
    two columns of a rotation matrix. The full rotation matrix is recovered
    using Gram-Schmidt orthonormalization.

    Attributes
    ----------
    vectors : Tensor
        First two columns of rotation matrix flattened, shape (..., 6).

    Examples
    --------
    Identity rotation (first two columns of I):
        Rotation6D(vectors=torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    Batch of 6D rotations:
        Rotation6D(vectors=torch.randn(100, 6))

    Notes
    -----
    The 6D representation has two key advantages:

    1. **Continuity**: Unlike quaternions or Euler angles, the 6D
       representation is continuous. This means small changes in rotation
       result in small changes in representation, which is important for
       neural network optimization.

    2. **No normalization constraint**: The representation does not require
       normalization during training, as Gram-Schmidt orthonormalization
       is applied when converting to a rotation matrix.

    References
    ----------
    .. [1] Zhou et al., "On the Continuity of Rotation Representations in
           Neural Networks", CVPR 2019.
    """

    vectors: Tensor


def rotation_6d(vectors: Tensor) -> Rotation6D:
    """Create 6D rotation from a tensor.

    Parameters
    ----------
    vectors : Tensor
        First two columns of rotation matrix flattened, shape (..., 6).

    Returns
    -------
    Rotation6D
        Rotation6D instance.

    Raises
    ------
    ValueError
        If vectors does not have last dimension 6.

    Examples
    --------
    >>> r6d = rotation_6d(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    >>> r6d.vectors
    tensor([1., 0., 0., 0., 1., 0.])
    """
    if vectors.shape[-1] != 6:
        raise ValueError(
            f"rotation_6d: vectors must have last dimension 6, got {vectors.shape[-1]}"
        )
    return Rotation6D(vectors=vectors)


def rotation_6d_to_matrix(r6d: Rotation6D) -> Tensor:
    """Convert 6D rotation to 3x3 rotation matrix.

    Uses Gram-Schmidt orthonormalization to convert the 6D representation
    (two potentially non-orthonormal vectors) into a proper rotation matrix.

    Parameters
    ----------
    r6d : Rotation6D
        6D rotation with shape (..., 6).

    Returns
    -------
    Tensor
        Rotation matrix, shape (..., 3, 3).

    Notes
    -----
    The Gram-Schmidt process:

    1. Normalize the first column: :math:`\\mathbf{b}_1 = \\mathbf{a}_1 / \\|\\mathbf{a}_1\\|`
    2. Remove component of second column along first:
       :math:`\\mathbf{b}_2 = \\mathbf{a}_2 - (\\mathbf{b}_1 \\cdot \\mathbf{a}_2) \\mathbf{b}_1`
    3. Normalize the second column: :math:`\\mathbf{b}_2 = \\mathbf{b}_2 / \\|\\mathbf{b}_2\\|`
    4. Compute third column via cross product: :math:`\\mathbf{b}_3 = \\mathbf{b}_1 \\times \\mathbf{b}_2`

    The resulting matrix :math:`[\\mathbf{b}_1, \\mathbf{b}_2, \\mathbf{b}_3]`
    is orthogonal with determinant 1.

    See Also
    --------
    matrix_to_rotation_6d : Inverse conversion.
    rotation_6d_to_quaternion : Convert to quaternion.

    References
    ----------
    .. [1] Zhou et al., "On the Continuity of Rotation Representations in
           Neural Networks", CVPR 2019.

    Examples
    --------
    Identity rotation:

    >>> r6d = rotation_6d(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    >>> rotation_6d_to_matrix(r6d)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])

    Non-orthonormal input is orthonormalized:

    >>> r6d = rotation_6d(torch.tensor([2.0, 0.0, 0.0, 1.0, 1.0, 0.0]))
    >>> R = rotation_6d_to_matrix(r6d)
    >>> R @ R.T  # Should be identity
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    """
    vectors = r6d.vectors

    # Split into first two columns
    a1 = vectors[..., :3]
    a2 = vectors[..., 3:]

    # Gram-Schmidt orthonormalization
    # Step 1: Normalize first column
    b1 = a1 / (torch.linalg.norm(a1, dim=-1, keepdim=True) + 1e-12)

    # Step 2: Remove component of a2 along b1 and normalize
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-12)

    # Step 3: Third column via cross product
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack columns to form rotation matrix
    # Shape: (..., 3, 3)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(matrix: Tensor) -> Rotation6D:
    """Convert 3x3 rotation matrix to 6D rotation.

    Extracts the first two columns of the rotation matrix and flattens them
    into a 6-dimensional vector.

    Parameters
    ----------
    matrix : Tensor
        Rotation matrix, shape (..., 3, 3).

    Returns
    -------
    Rotation6D
        6D rotation with shape (..., 6).

    Notes
    -----
    This conversion simply extracts the first two columns of the rotation
    matrix. The result can be used to reconstruct the full matrix via
    Gram-Schmidt orthonormalization.

    See Also
    --------
    rotation_6d_to_matrix : Inverse conversion.
    quaternion_to_rotation_6d : Convert from quaternion.

    Examples
    --------
    Identity matrix:

    >>> R = torch.eye(3)
    >>> r6d = matrix_to_rotation_6d(R)
    >>> r6d.vectors
    tensor([1., 0., 0., 0., 1., 0.])

    Round-trip conversion:

    >>> r6d = rotation_6d(torch.randn(6))
    >>> R = rotation_6d_to_matrix(r6d)
    >>> r6d_back = matrix_to_rotation_6d(R)
    >>> torch.allclose(rotation_6d_to_matrix(r6d_back), R)
    True
    """
    # Extract first two columns
    col1 = matrix[..., :, 0]  # Shape: (..., 3)
    col2 = matrix[..., :, 1]  # Shape: (..., 3)

    # Flatten into 6D vector
    vectors = torch.cat([col1, col2], dim=-1)

    return Rotation6D(vectors=vectors)


def rotation_6d_to_quaternion(r6d: Rotation6D) -> Quaternion:
    """Convert 6D rotation to quaternion.

    Converts via rotation matrix representation.

    Parameters
    ----------
    r6d : Rotation6D
        6D rotation with shape (..., 6).

    Returns
    -------
    Quaternion
        Unit quaternion in wxyz convention, shape (..., 4).

    Notes
    -----
    - Converts to matrix first, then to quaternion.
    - The output quaternion is always a unit quaternion.

    See Also
    --------
    quaternion_to_rotation_6d : Inverse conversion.
    rotation_6d_to_matrix : Convert to matrix directly.

    Examples
    --------
    Identity rotation:

    >>> r6d = rotation_6d(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    >>> rotation_6d_to_quaternion(r6d).wxyz
    tensor([1., 0., 0., 0.])
    """
    matrix = rotation_6d_to_matrix(r6d)
    return matrix_to_quaternion(matrix)


def quaternion_to_rotation_6d(q: Quaternion) -> Rotation6D:
    """Convert quaternion to 6D rotation.

    Converts via rotation matrix representation.

    Parameters
    ----------
    q : Quaternion
        Unit quaternion in wxyz convention, shape (..., 4).

    Returns
    -------
    Rotation6D
        6D rotation with shape (..., 6).

    Notes
    -----
    - Converts to matrix first, then extracts first two columns.
    - The output represents the same rotation as the input quaternion.

    See Also
    --------
    rotation_6d_to_quaternion : Inverse conversion.
    matrix_to_rotation_6d : Convert from matrix directly.

    Examples
    --------
    Identity quaternion:

    >>> from torchscience.geometry.transform import quaternion
    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> quaternion_to_rotation_6d(q).vectors
    tensor([1., 0., 0., 0., 1., 0.])
    """
    matrix = quaternion_to_matrix(q)
    return matrix_to_rotation_6d(matrix)

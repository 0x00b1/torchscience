"""RotationMatrix representation."""

from __future__ import annotations

from tensordict.tensorclass import tensorclass
from torch import Tensor


@tensorclass
class RotationMatrix:
    """3x3 rotation matrix (SO(3) element).

    A rotation matrix R in SO(3) satisfies:
    - Orthogonality: R^T R = I
    - Unit determinant: det(R) = +1

    Attributes
    ----------
    matrix : Tensor
        Rotation matrix, shape (..., 3, 3).

    Examples
    --------
    Identity rotation:
        RotationMatrix(matrix=torch.eye(3))

    Batch of rotation matrices:
        RotationMatrix(matrix=torch.randn(100, 3, 3))

    Notes
    -----
    This class does not enforce orthogonality or unit determinant.
    Use :func:`quaternion_to_matrix` to create valid rotation matrices
    from unit quaternions.
    """

    matrix: Tensor


def rotation_matrix(matrix: Tensor) -> RotationMatrix:
    """Create RotationMatrix from matrix tensor.

    Parameters
    ----------
    matrix : Tensor
        Rotation matrix, shape (..., 3, 3).

    Returns
    -------
    RotationMatrix
        RotationMatrix instance.

    Raises
    ------
    ValueError
        If matrix does not have last two dimensions (3, 3).

    Examples
    --------
    >>> R = rotation_matrix(torch.eye(3))
    >>> R.matrix
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotation_matrix: matrix must have last two dimensions (3, 3), got {matrix.shape[-2:]}"
        )
    return RotationMatrix(matrix=matrix)

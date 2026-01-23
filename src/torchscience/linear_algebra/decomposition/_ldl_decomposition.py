"""LDL decomposition for symmetric/Hermitian matrices."""

import torch
from torch import Tensor

from torchscience.linear_algebra.decomposition._result_types import (
    LDLDecompositionResult,
)


def ldl_decomposition(a: Tensor) -> LDLDecompositionResult:
    r"""
    LDL decomposition for symmetric/Hermitian matrices.

    Computes the LDL decomposition:

    .. math::

        A = L D L^*

    where :math:`L` is unit lower triangular (ones on the diagonal),
    :math:`D` is diagonal, and :math:`L^*` is the conjugate transpose of
    :math:`L` (for real matrices, :math:`L^* = L^T`).

    Parameters
    ----------
    a : Tensor
        Symmetric (for real) or Hermitian (for complex) input matrix of shape
        (..., n, n). Must be a floating-point or complex tensor.
        Only the lower triangular part is used.

    Returns
    -------
    LDLDecompositionResult
        A named tuple containing:

        - **L** (*Tensor*) - Unit lower triangular matrix of shape (..., n, n).
          Has ones on the diagonal.
        - **D** (*Tensor*) - Diagonal matrix of shape (..., n, n). For
          positive definite matrices, the diagonal entries are positive.
          For indefinite matrices, entries may be negative.
        - **pivots** (*Tensor*) - Pivot indices of shape (..., n). These describe
          the symmetric permutation applied during factorization for numerical
          stability.
        - **info** (*Tensor*) - Integer tensor of shape (...). A value of 0
          indicates successful computation.

    Raises
    ------
    ValueError
        If input is not at least 2D or is not square.

    Notes
    -----
    The LDL decomposition is a variant of Cholesky decomposition that does not
    require the matrix to be positive definite. It is numerically stable for
    symmetric indefinite matrices when combined with pivoting (Bunch-Kaufman
    pivoting).

    For a symmetric positive definite matrix, the Cholesky factor :math:`C` is
    related to the LDL factors by :math:`C = L \sqrt{D}`, where :math:`\sqrt{D}`
    is the element-wise square root of the diagonal.

    To reconstruct the original matrix from the factors:

    .. code-block:: python

        # For real symmetric matrices
        A_reconstructed = L @ D @ L.T

        # For complex Hermitian matrices
        A_reconstructed = L @ D @ L.mH

    The input matrix should satisfy :math:`A = A^T` (symmetric) or
    :math:`A = A^H` (Hermitian). Only the lower triangular part of the input
    is used for computation.

    Examples
    --------
    Decompose a symmetric positive definite matrix:

    >>> import torch
    >>> from torchscience.linear_algebra.decomposition import ldl_decomposition
    >>> a = torch.tensor([[4., 2.], [2., 5.]], dtype=torch.float64)
    >>> result = ldl_decomposition(a)
    >>> result.L  # Unit lower triangular
    >>> result.D  # Diagonal matrix with positive entries

    Verify reconstruction:

    >>> reconstructed = result.L @ result.D @ result.L.T
    >>> torch.allclose(reconstructed, a)
    True

    Decompose a symmetric indefinite matrix:

    >>> a = torch.tensor([[1., 2.], [2., 1.]], dtype=torch.float64)
    >>> result = ldl_decomposition(a)
    >>> # D may have negative diagonal entries for indefinite matrices
    >>> torch.diag(result.D)  # Shows the diagonal entries

    See Also
    --------
    torch.linalg.cholesky : Cholesky decomposition for positive definite matrices
    torch.linalg.ldl_factor : PyTorch's underlying LDL factorization
    """
    if a.dim() < 2:
        raise ValueError(
            f"ldl_decomposition: a must be at least 2D, got {a.dim()}D"
        )

    if a.size(-2) != a.size(-1):
        raise ValueError(
            f"ldl_decomposition: a must be square, got shape {tuple(a.shape[-2:])}"
        )

    L, D, pivots, info = torch.ops.torchscience.ldl_decomposition(a)

    return LDLDecompositionResult(L=L, D=D, pivots=pivots, info=info)

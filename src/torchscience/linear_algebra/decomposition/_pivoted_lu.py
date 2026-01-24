"""Pivoted LU decomposition."""

import torch
from torch import Tensor

from torchscience.linear_algebra.decomposition._result_types import (
    PivotedLUResult,
)


def pivoted_lu(a: Tensor) -> PivotedLUResult:
    r"""
    Pivoted LU decomposition.

    Computes the LU decomposition with row pivoting:

    .. math::

        PA = LU

    where :math:`P` is a permutation matrix, :math:`L` is unit lower triangular
    (ones on the diagonal), and :math:`U` is upper triangular.

    Parameters
    ----------
    a : Tensor
        Input matrix of shape (..., m, n). Must be a floating-point or complex
        tensor.

    Returns
    -------
    PivotedLUResult
        A named tuple containing:

        - **L** (*Tensor*) - Unit lower triangular matrix of shape (..., m, k)
          where k = min(m, n). Has ones on the diagonal.
        - **U** (*Tensor*) - Upper triangular matrix of shape (..., k, n).
        - **pivots** (*Tensor*) - Pivot indices of shape (..., k). These describe
          the permutation matrix P: pivots[i] indicates that row i was swapped
          with row pivots[i] during factorization.
        - **info** (*Tensor*) - Integer tensor of shape (...). A value of 0
          indicates successful computation.

    Raises
    ------
    ValueError
        If input is not at least 2D.

    Notes
    -----
    The LU decomposition with partial pivoting factors a matrix :math:`A` as
    :math:`PA = LU` where row interchanges are performed to ensure numerical
    stability. This is the standard form used in numerical linear algebra.

    To reconstruct the original matrix from the factors:

    1. Construct the permutation matrix P from the pivot indices
    2. Compute :math:`A = P^T L U` (since :math:`PA = LU` means :math:`A = P^{-1}LU = P^T LU`)

    For a square matrix, L and U both have shape (n, n). For rectangular matrices:

    - If m > n (tall): L is (m, n), U is (n, n)
    - If m < n (wide): L is (m, m), U is (m, n)

    Examples
    --------
    >>> import torch
    >>> from torchscience.linear_algebra.decomposition import pivoted_lu
    >>> a = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    >>> result = pivoted_lu(a)
    >>> result.L  # Unit lower triangular
    >>> result.U  # Upper triangular
    >>> result.pivots  # Pivot indices

    Verify reconstruction using pivots:

    >>> m, n = a.shape
    >>> P = torch.zeros(m, m, dtype=torch.float64)
    >>> P[torch.arange(m), result.pivots[:m]] = 1
    >>> torch.allclose(P.T @ result.L @ result.U, a)
    True
    """
    if a.dim() < 2:
        raise ValueError(f"pivoted_lu: a must be at least 2D, got {a.dim()}D")

    L, U, pivots, info = torch.ops.torchscience.pivoted_lu(a)

    return PivotedLUResult(L=L, U=U, pivots=pivots, info=info)

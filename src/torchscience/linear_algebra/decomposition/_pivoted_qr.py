"""Pivoted QR decomposition."""

import torch
from torch import Tensor

from torchscience.linear_algebra.decomposition._result_types import (
    PivotedQRResult,
)


def pivoted_qr(a: Tensor) -> PivotedQRResult:
    r"""
    Pivoted QR decomposition with column pivoting.

    Computes the QR decomposition with column pivoting:

    .. math::

        AP = QR

    where :math:`P` is a column permutation matrix, :math:`Q` is an orthogonal
    (or unitary for complex inputs) matrix, and :math:`R` is upper triangular.

    Equivalently, this can be written as:

    .. math::

        A[:, \text{pivots}] = QR

    Column pivoting selects columns with maximum remaining norm at each step,
    which improves numerical stability and provides rank-revealing properties.

    Parameters
    ----------
    a : Tensor
        Input matrix of shape (..., m, n). Must be a floating-point or complex
        tensor.

    Returns
    -------
    PivotedQRResult
        A named tuple containing:

        - **Q** (*Tensor*) - Orthogonal/unitary matrix of shape (..., m, k)
          where k = min(m, n). Satisfies Q^H Q = I.
        - **R** (*Tensor*) - Upper triangular matrix of shape (..., k, n).
          The diagonal elements are ordered by decreasing magnitude (when
          column pivoting is effective).
        - **pivots** (*Tensor*) - Column permutation indices of shape (..., n).
          These describe how columns were reordered: A[:, pivots] = Q @ R.
        - **info** (*Tensor*) - Integer tensor of shape (...). A value of 0
          indicates successful computation.

    Raises
    ------
    ValueError
        If input is not at least 2D.

    Notes
    -----
    The pivoted QR decomposition is useful for:

    1. **Numerical stability**: Column pivoting ensures that the diagonal
       elements of R are ordered by magnitude, reducing sensitivity to
       round-off errors.

    2. **Rank determination**: For rank-deficient matrices, the diagonal
       of R will have small elements at the end, allowing estimation of
       the numerical rank.

    3. **Least squares**: Provides a more stable solution than standard QR
       for ill-conditioned systems.

    To reconstruct the original matrix from the factors:

    .. code-block:: python

        # Method 1: Direct indexing
        A_reconstructed = Q @ R
        A[:, pivots] == A_reconstructed  # True (approximately)

        # Method 2: Using inverse permutation
        inv_pivots = torch.argsort(pivots)
        A == (Q @ R)[:, inv_pivots]  # True (approximately)

    For a square matrix, Q has shape (n, n) and R has shape (n, n).
    For rectangular matrices:

    - If m > n (tall): Q is (m, n), R is (n, n)
    - If m < n (wide): Q is (m, m), R is (m, n)

    Examples
    --------
    >>> import torch
    >>> from torchscience.linear_algebra.decomposition import pivoted_qr
    >>> a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
    >>> result = pivoted_qr(a)
    >>> result.Q.shape  # (2, 2) since k = min(2, 3) = 2
    torch.Size([2, 2])
    >>> result.R.shape  # (2, 3)
    torch.Size([2, 3])
    >>> result.pivots  # Column permutation
    tensor([...])

    Verify reconstruction:

    >>> A_permuted = a[:, result.pivots]
    >>> torch.allclose(A_permuted, result.Q @ result.R)
    True

    Verify Q is orthogonal:

    >>> torch.allclose(result.Q.T @ result.Q, torch.eye(2, dtype=torch.float64))
    True
    """
    if a.dim() < 2:
        raise ValueError(f"pivoted_qr: a must be at least 2D, got {a.dim()}D")

    Q, R, pivots, info = torch.ops.torchscience.pivoted_qr(a)

    return PivotedQRResult(Q=Q, R=R, pivots=pivots, info=info)

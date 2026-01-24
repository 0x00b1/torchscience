"""Rank-revealing QR decomposition."""

import torch
from torch import Tensor

from torchscience.linear_algebra.decomposition._result_types import (
    RankRevealingQRResult,
)


def rank_revealing_qr(
    a: Tensor,
    *,
    tol: float = 1e-10,
) -> RankRevealingQRResult:
    r"""
    Rank-revealing QR decomposition with column pivoting.

    Computes the QR decomposition with column pivoting and numerical rank detection:

    .. math::

        AP = QR

    where :math:`P` is a column permutation matrix, :math:`Q` is an orthogonal
    (or unitary for complex inputs) matrix, and :math:`R` is upper triangular.

    The numerical rank is determined by counting the number of diagonal elements
    of :math:`R` that satisfy :math:`|R_{ii}| > \text{tol} \cdot |R_{00}|`.

    Parameters
    ----------
    a : Tensor
        Input matrix of shape (..., m, n). Must be a floating-point or complex
        tensor.
    tol : float, optional
        Tolerance for rank determination. The numerical rank is the count of
        diagonal elements of R satisfying |R[i,i]| > tol * |R[0,0]|.
        Default is 1e-10.

    Returns
    -------
    RankRevealingQRResult
        A named tuple containing:

        - **Q** (*Tensor*) - Orthogonal/unitary matrix of shape (..., m, k)
          where k = min(m, n). Satisfies Q^H Q = I.
        - **R** (*Tensor*) - Upper triangular matrix of shape (..., k, n).
          The diagonal elements are ordered by decreasing magnitude.
        - **pivots** (*Tensor*) - Column permutation indices of shape (..., n).
          These describe how columns were reordered: A[:, pivots] = Q @ R.
        - **rank** (*Tensor*) - Numerical rank tensor of shape (...).
          The estimated rank based on the tolerance criterion.
        - **info** (*Tensor*) - Integer tensor of shape (...). A value of 0
          indicates successful computation.

    Raises
    ------
    ValueError
        If input is not at least 2D.

    Notes
    -----
    The rank-revealing QR decomposition extends pivoted QR with automatic
    numerical rank detection. This is useful for:

    1. **Rank determination**: Automatically estimates the numerical rank
       of a matrix, which is robust to noise and round-off errors.

    2. **Ill-conditioned systems**: Identifies when a system is numerically
       rank-deficient, allowing for appropriate handling.

    3. **Low-rank approximations**: The rank output directly indicates
       how many columns of Q and rows of R are significant.

    4. **Regularization**: The rank can guide regularization strategies
       in least-squares problems.

    The tolerance parameter controls the sensitivity of rank detection:
    - Smaller tol values yield higher rank estimates (less filtering)
    - Larger tol values yield lower rank estimates (more filtering)
    - A reasonable default is around machine epsilon times the matrix dimension

    To reconstruct a low-rank approximation:

    .. code-block:: python

        result = rank_revealing_qr(A)
        r = result.rank.item()  # numerical rank
        Q_r = result.Q[:, :r]   # truncate to rank-r
        R_r = result.R[:r, :]   # truncate to rank-r
        # Low-rank approximation: Q_r @ R_r (permuted)

    Examples
    --------
    >>> import torch
    >>> from torchscience.linear_algebra.decomposition import rank_revealing_qr
    >>> a = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=torch.float64)
    >>> result = rank_revealing_qr(a)
    >>> result.rank  # This matrix has rank 2
    tensor(2)

    Test with a full-rank matrix:

    >>> a_full = torch.eye(3, dtype=torch.float64)
    >>> result_full = rank_revealing_qr(a_full)
    >>> result_full.rank
    tensor(3)

    Test with a zero matrix:

    >>> a_zero = torch.zeros(3, 3, dtype=torch.float64)
    >>> result_zero = rank_revealing_qr(a_zero)
    >>> result_zero.rank
    tensor(0)

    Verify reconstruction:

    >>> a = torch.randn(4, 4, dtype=torch.float64)
    >>> result = rank_revealing_qr(a)
    >>> A_permuted = a[:, result.pivots]
    >>> torch.allclose(A_permuted, result.Q @ result.R)
    True
    """
    if a.dim() < 2:
        raise ValueError(
            f"rank_revealing_qr: a must be at least 2D, got {a.dim()}D"
        )

    if tol < 0:
        raise ValueError(
            f"rank_revealing_qr: tol must be non-negative, got {tol}"
        )

    Q, R, pivots, rank, info = torch.ops.torchscience.rank_revealing_qr(a, tol)

    return RankRevealingQRResult(Q=Q, R=R, pivots=pivots, rank=rank, info=info)

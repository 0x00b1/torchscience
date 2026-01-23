"""Jordan decomposition."""

import torch
from torch import Tensor

from torchscience.linear_algebra.decomposition._result_types import (
    JordanDecompositionResult,
)


def jordan_decomposition(a: Tensor) -> JordanDecompositionResult:
    r"""
    Jordan decomposition.

    Computes the Jordan decomposition :math:`A = PJP^{-1}` where :math:`J` is
    the Jordan normal form and :math:`P` is the similarity transformation matrix.

    For generic (diagonalizable) matrices, :math:`J` is a diagonal matrix with
    eigenvalues on the diagonal. For defective matrices (matrices that are not
    diagonalizable), :math:`J` contains Jordan blocks.

    .. note::
        This function does not support gradients as the Jordan form is
        discontinuous with respect to matrix entries. The function detaches
        inputs from the computation graph.

    .. warning::
        This is a simplified implementation that computes the Jordan form
        for diagonalizable matrices. For defective matrices (repeated eigenvalues
        with insufficient eigenvectors), the decomposition may be numerically
        unstable or incorrect. The ``info`` tensor indicates potential issues.

    Parameters
    ----------
    a : Tensor
        Input matrix of shape (..., n, n). Must be a floating-point or complex
        tensor.

    Returns
    -------
    JordanDecompositionResult
        A named tuple containing:

        - **J** (*Tensor*) - Jordan normal form of shape (..., n, n). Always
          complex dtype. For diagonalizable matrices, this is diagonal with
          eigenvalues on the diagonal.
        - **P** (*Tensor*) - Similarity transformation matrix of shape (..., n, n).
          Always complex dtype. Contains (generalized) eigenvectors.
        - **info** (*Tensor*) - Integer tensor of shape (...). Values:
          - 0: successful computation
          - 1: matrix may be nearly defective (ill-conditioned eigenvectors)
          - -1: computation failed

    Raises
    ------
    ValueError
        If input is not at least 2D or not square.

    Notes
    -----
    The Jordan normal form is a canonical form for a matrix over an algebraically
    closed field. Every square matrix :math:`A` can be written as
    :math:`A = PJP^{-1}` where :math:`J` is block diagonal with Jordan blocks.

    A Jordan block for eigenvalue :math:`\lambda` of size :math:`k` is:

    .. math::

        J_k(\lambda) = \begin{pmatrix}
            \lambda & 1 & 0 & \cdots & 0 \\
            0 & \lambda & 1 & \cdots & 0 \\
            \vdots & & \ddots & \ddots & \vdots \\
            0 & 0 & \cdots & \lambda & 1 \\
            0 & 0 & \cdots & 0 & \lambda
        \end{pmatrix}

    For diagonalizable matrices (the generic case), all Jordan blocks have size 1,
    so :math:`J` is simply a diagonal matrix of eigenvalues.

    The eigenvalues in :math:`J` appear in the same order as the columns of :math:`P`,
    which are the corresponding (generalized) eigenvectors.

    Examples
    --------
    >>> import torch
    >>> from torchscience.linear_algebra.decomposition import jordan_decomposition
    >>> a = torch.tensor([[1., 2.], [3., 4.]])
    >>> result = jordan_decomposition(a)
    >>> result.J  # Jordan form (diagonal for this diagonalizable matrix)
    >>> result.P  # Eigenvector matrix
    >>> # Verify: A = P @ J @ P^{-1}
    >>> reconstructed = result.P @ result.J @ torch.linalg.inv(result.P)
    >>> torch.allclose(reconstructed.real, a, atol=1e-6)
    True

    >>> # With a diagonal matrix (already in Jordan form)
    >>> d = torch.diag(torch.tensor([1., 2., 3.]))
    >>> result = jordan_decomposition(d)
    >>> torch.allclose(torch.diag(result.J).real, torch.tensor([1., 2., 3.]), atol=1e-6)
    True
    """
    if a.dim() < 2:
        raise ValueError(f"a must be at least 2D, got {a.dim()}D")
    if a.shape[-2] != a.shape[-1]:
        raise ValueError(f"a must be square, got shape {a.shape}")

    # Jordan decomposition is discontinuous, so we don't track gradients
    a_detached = a.detach()

    J, P, info = torch.ops.torchscience.jordan_decomposition(a_detached)

    return JordanDecompositionResult(J=J, P=P, info=info)

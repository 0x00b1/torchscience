"""Hessenberg decomposition."""

import torch
from torch import Tensor

from torchscience.linear_algebra.decomposition._result_types import (
    HessenbergResult,
)


def hessenberg(a: Tensor) -> HessenbergResult:
    r"""
    Hessenberg decomposition.

    Computes the Hessenberg decomposition :math:`A = QHQ^*` where :math:`H` is
    upper Hessenberg (has zeros below the first subdiagonal) and :math:`Q` is
    unitary.

    The upper Hessenberg form is useful as an intermediate step in eigenvalue
    algorithms since it preserves eigenvalues while having a simpler structure
    than a general matrix.

    Parameters
    ----------
    a : Tensor
        Input matrix of shape (..., n, n). Must be a floating-point or complex
        tensor.

    Returns
    -------
    HessenbergResult
        A named tuple containing:

        - **H** (*Tensor*) - Upper Hessenberg matrix of shape (..., n, n).
          Has zeros below the first subdiagonal.
        - **Q** (*Tensor*) - Unitary transformation matrix of shape (..., n, n).
          Satisfies :math:`Q Q^* = Q^* Q = I`.
        - **info** (*Tensor*) - Integer tensor of shape (...). A value of 0
          indicates successful computation.

    Raises
    ------
    ValueError
        If input is not at least 2D or not square.

    Notes
    -----
    The decomposition is computed using Householder reflections. For a matrix
    :math:`A` of size :math:`n \times n`, the algorithm applies :math:`n-2`
    Householder transformations to reduce :math:`A` to upper Hessenberg form.

    An upper Hessenberg matrix :math:`H` satisfies :math:`H_{ij} = 0` for
    :math:`i > j + 1`. That is, all entries below the first subdiagonal are zero.

    The decomposition satisfies:

    .. math::

        A = Q H Q^*

    where :math:`Q^*` denotes the conjugate transpose of :math:`Q`.

    Examples
    --------
    >>> import torch
    >>> from torchscience.linear_algebra.decomposition import hessenberg
    >>> a = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> result = hessenberg(a)
    >>> result.H  # Upper Hessenberg form
    >>> result.Q  # Unitary matrix
    >>> torch.allclose(result.Q @ result.H @ result.Q.mH, a)  # Verify reconstruction
    True
    """
    if a.dim() < 2:
        raise ValueError(f"a must be at least 2D, got {a.dim()}D")
    if a.shape[-2] != a.shape[-1]:
        raise ValueError(f"a must be square, got shape {a.shape}")

    H, Q, info = torch.ops.torchscience.hessenberg(a)

    return HessenbergResult(H=H, Q=Q, info=info)

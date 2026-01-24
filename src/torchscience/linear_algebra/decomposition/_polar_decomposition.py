"""Polar decomposition."""

import torch
from torch import Tensor

from torchscience.linear_algebra.decomposition._result_types import (
    PolarDecompositionResult,
)


def polar_decomposition(
    a: Tensor,
    *,
    side: str = "right",
) -> PolarDecompositionResult:
    r"""
    Polar decomposition.

    Computes the polar decomposition of a matrix :math:`A` into the product of a
    unitary matrix :math:`U` and a positive semidefinite Hermitian matrix :math:`P`.

    For the right polar decomposition:

    .. math::

        A = UP

    where :math:`P = (A^*A)^{1/2}` and :math:`U = A P^{-1}`.

    For the left polar decomposition:

    .. math::

        A = PU

    where :math:`P = (AA^*)^{1/2}` and :math:`U = P^{-1} A`.

    The decomposition is computed via the SVD. If :math:`A = U_{svd} \Sigma V^*` is
    the SVD, then:

    - Right polar: :math:`U = U_{svd} V^*` and :math:`P = V \Sigma V^*`
    - Left polar: :math:`U = U_{svd} V^*` and :math:`P = U_{svd} \Sigma U_{svd}^*`

    Parameters
    ----------
    a : Tensor
        Input matrix of shape (..., m, n). Must be a floating-point or complex
        tensor.
    side : str, optional
        Which polar decomposition to compute:

        - ``'right'`` (default): Compute :math:`A = UP` where :math:`P` has shape
          (..., n, n).
        - ``'left'``: Compute :math:`A = PU` where :math:`P` has shape (..., m, m).

    Returns
    -------
    PolarDecompositionResult
        A named tuple containing:

        - **U** (*Tensor*) - Unitary matrix of shape (..., m, n). Satisfies
          :math:`U U^* = I` when :math:`m \leq n` (right polar) or
          :math:`U^* U = I` when :math:`m \geq n` (left polar).
        - **P** (*Tensor*) - Positive semidefinite Hermitian matrix. Has shape
          (..., n, n) for right polar or (..., m, m) for left polar.
        - **info** (*Tensor*) - Integer tensor of shape (...). A value of 0
          indicates successful computation.

    Raises
    ------
    ValueError
        If input is not at least 2D, or if side is not 'right' or 'left'.

    Notes
    -----
    The polar decomposition is a generalization of the polar form of a complex
    number :math:`z = e^{i\theta} r` to matrices.

    For a square, invertible matrix, the polar decomposition is unique. For
    rank-deficient matrices, :math:`U` is not unique but :math:`P` is always
    unique.

    The positive semidefinite factor :math:`P` satisfies:

    - :math:`P = P^*` (Hermitian)
    - All eigenvalues of :math:`P` are non-negative
    - :math:`P^2 = A^* A` (right) or :math:`P^2 = A A^*` (left)

    Examples
    --------
    >>> import torch
    >>> from torchscience.linear_algebra.decomposition import polar_decomposition
    >>> a = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    >>> result = polar_decomposition(a)
    >>> result.U  # Unitary factor
    >>> result.P  # Positive semidefinite factor
    >>> torch.allclose(result.U @ result.P, a)  # Verify reconstruction
    True

    Left polar decomposition:

    >>> result_left = polar_decomposition(a, side='left')
    >>> torch.allclose(result_left.P @ result_left.U, a)  # Verify A = PU
    True
    """
    if a.dim() < 2:
        raise ValueError(f"a must be at least 2D, got {a.dim()}D")
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got '{side}'")

    U, P, info = torch.ops.torchscience.polar_decomposition(a, side)

    return PolarDecompositionResult(U=U, P=P, info=info)

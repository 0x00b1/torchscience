"""Generalized Schur (QZ) decomposition."""

import torch
from torch import Tensor

from torchscience.linear_algebra.decomposition._result_types import (
    GeneralizedSchurResult,
)


def generalized_schur(
    a: Tensor,
    b: Tensor,
    *,
    output: str = "real",
) -> GeneralizedSchurResult:
    r"""
    Generalized Schur (QZ) decomposition.

    Computes the generalized Schur decomposition of matrix pencil (A, B):

    .. math::

        A = Q S Z^H \\
        B = Q T Z^H

    where S and T are upper triangular (complex output) or quasi-upper-triangular
    (real output), and Q and Z are unitary matrices.

    The generalized eigenvalues of the pencil (A, B) are given by
    :math:`\lambda_i = \alpha_i / \beta_i` where :math:`\alpha_i = S_{ii}`
    and :math:`\beta_i = T_{ii}`.

    Parameters
    ----------
    a : Tensor
        First input matrix of shape (..., n, n).
    b : Tensor
        Second input matrix of shape (..., n, n). Must be the same size as a.
    output : str, optional
        Output form: 'real' for quasi-upper-triangular form (default),
        'complex' for strictly upper triangular form.

    Returns
    -------
    GeneralizedSchurResult
        Named tuple containing:

        - S : Tensor of shape (..., n, n)
            Schur form of A (upper triangular or quasi-upper-triangular).
        - T : Tensor of shape (..., n, n)
            Schur form of B (upper triangular).
        - alpha : Tensor of shape (..., n), complex
            Eigenvalue numerators. Generalized eigenvalues are alpha/beta.
        - beta : Tensor of shape (..., n), complex
            Eigenvalue denominators. Generalized eigenvalues are alpha/beta.
        - Q : Tensor of shape (..., n, n)
            Left unitary transformation matrix.
        - Z : Tensor of shape (..., n, n)
            Right unitary transformation matrix.
        - info : Tensor of shape (...), int
            Convergence info. 0 indicates success.

    Examples
    --------
    >>> import torch
    >>> from torchscience.linear_algebra.decomposition import generalized_schur
    >>> a = torch.randn(3, 3, dtype=torch.float64)
    >>> b = torch.randn(3, 3, dtype=torch.float64)
    >>> result = generalized_schur(a, b, output="complex")
    >>> # Verify A = Q @ S @ Z.H
    >>> reconstructed_a = result.Q @ result.S @ result.Z.mH
    >>> torch.allclose(reconstructed_a, a.to(torch.complex128), atol=1e-10)
    True
    >>> # Verify B = Q @ T @ Z.H
    >>> reconstructed_b = result.Q @ result.T @ result.Z.mH
    >>> torch.allclose(reconstructed_b, b.to(torch.complex128), atol=1e-10)
    True

    Notes
    -----
    The generalized Schur decomposition (also known as QZ decomposition) is
    useful for solving generalized eigenvalue problems without explicitly
    inverting B, which can be numerically unstable when B is near-singular.

    When beta[i] = 0, the corresponding generalized eigenvalue is infinite,
    indicating that B has a zero eigenvalue in that direction.
    """
    if a.dim() < 2:
        raise ValueError(f"a must be at least 2D, got {a.dim()}D")
    if b.dim() < 2:
        raise ValueError(f"b must be at least 2D, got {b.dim()}D")
    if a.shape[-2] != a.shape[-1]:
        raise ValueError(f"a must be square, got shape {a.shape}")
    if b.shape[-2] != b.shape[-1]:
        raise ValueError(f"b must be square, got shape {b.shape}")
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"a and b must have the same size, got a: {a.shape} and b: {b.shape}"
        )
    if output not in ("real", "complex"):
        raise ValueError(f"output must be 'real' or 'complex', got '{output}'")

    S, T, alpha, beta, Q, Z, info = torch.ops.torchscience.generalized_schur(
        a, b, output
    )

    return GeneralizedSchurResult(
        S=S,
        T=T,
        alpha=alpha,
        beta=beta,
        Q=Q,
        Z=Z,
        info=info,
    )

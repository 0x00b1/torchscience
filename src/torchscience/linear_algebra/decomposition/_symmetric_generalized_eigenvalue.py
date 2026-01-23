"""Symmetric generalized eigenvalue decomposition."""

import torch
from torch import Tensor

from torchscience.linear_algebra.decomposition._result_types import (
    SymmetricGeneralizedEigenvalueResult,
)


def symmetric_generalized_eigenvalue(
    a: Tensor,
    b: Tensor,
) -> SymmetricGeneralizedEigenvalueResult:
    r"""
    Symmetric generalized eigenvalue decomposition.

    Computes eigenvalues and eigenvectors for Ax = λBx where A is symmetric
    and B is symmetric positive definite.

    The eigenvalues are real and the eigenvectors are B-orthonormal:
    V^T B V = I.

    Parameters
    ----------
    a : Tensor
        Symmetric input matrix of shape (..., n, n).
    b : Tensor
        Symmetric positive definite matrix of shape (..., n, n).

    Returns
    -------
    SymmetricGeneralizedEigenvalueResult
        eigenvalues : Tensor of shape (..., n), real, sorted ascending
        eigenvectors : Tensor of shape (..., n, n), columns are eigenvectors
        info : Tensor of shape (...), int, 0 indicates success
    """
    # Input validation
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
            f"a and b must have same size, got {a.shape[-1]} and {b.shape[-1]}"
        )

    eigenvalues, eigenvectors, info = (
        torch.ops.torchscience.symmetric_generalized_eigenvalue(a, b)
    )

    return SymmetricGeneralizedEigenvalueResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        info=info,
    )

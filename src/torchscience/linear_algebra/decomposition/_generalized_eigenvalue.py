"""Generalized eigenvalue decomposition."""

import torch
from torch import Tensor

from torchscience.linear_algebra.decomposition._result_types import (
    GeneralizedEigenvalueResult,
)


def generalized_eigenvalue(
    a: Tensor,
    b: Tensor,
) -> GeneralizedEigenvalueResult:
    r"""
    Generalized eigenvalue decomposition.

    Computes eigenvalues λ and eigenvectors for Ax = λBx where A and B
    are general square matrices.

    Parameters
    ----------
    a : Tensor
        Input matrix of shape (..., n, n).
    b : Tensor
        Input matrix of shape (..., n, n).

    Returns
    -------
    GeneralizedEigenvalueResult
        eigenvalues : Tensor of shape (..., n), complex
        eigenvectors_left : Tensor of shape (..., n, n), complex
        eigenvectors_right : Tensor of shape (..., n, n), complex
        info : Tensor of shape (...), int, 0 indicates success
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
            f"a and b must have same size, got {a.shape[-1]} and {b.shape[-1]}"
        )

    eigenvalues, eigenvectors_left, eigenvectors_right, info = (
        torch.ops.torchscience.generalized_eigenvalue(a, b)
    )

    return GeneralizedEigenvalueResult(
        eigenvalues=eigenvalues,
        eigenvectors_left=eigenvectors_left,
        eigenvectors_right=eigenvectors_right,
        info=info,
    )

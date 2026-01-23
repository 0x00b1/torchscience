"""Schur decomposition."""

import torch
from torch import Tensor

from torchscience.linear_algebra.decomposition._result_types import (
    SchurDecompositionResult,
)


def schur_decomposition(
    a: Tensor,
    *,
    output: str = "real",
) -> SchurDecompositionResult:
    r"""
    Schur decomposition.

    Computes the Schur decomposition A = QTQ* where Q is unitary and T
    is upper triangular (complex) or quasi-upper-triangular (real).

    Parameters
    ----------
    a : Tensor
        Input matrix of shape (..., n, n).
    output : str
        'real' for real Schur form (quasi-triangular), 'complex' for
        complex Schur form (strictly triangular).

    Returns
    -------
    SchurDecompositionResult
        T : Tensor of shape (..., n, n)
        Q : Tensor of shape (..., n, n)
        eigenvalues : Tensor of shape (..., n), complex
        info : Tensor of shape (...), int
    """
    if a.dim() < 2:
        raise ValueError(f"a must be at least 2D, got {a.dim()}D")
    if a.shape[-2] != a.shape[-1]:
        raise ValueError(f"a must be square, got shape {a.shape}")
    if output not in ("real", "complex"):
        raise ValueError(f"output must be 'real' or 'complex', got '{output}'")

    T, Q, eigenvalues, info = torch.ops.torchscience.schur_decomposition(
        a, output
    )

    return SchurDecompositionResult(
        T=T,
        Q=Q,
        eigenvalues=eigenvalues,
        info=info,
    )

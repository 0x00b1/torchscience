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

    # Promote to common dtype
    dtype = torch.promote_types(a.dtype, b.dtype)
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float64
    a = a.to(dtype)
    b = b.to(dtype)

    # Broadcast batch dimensions
    batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = a.expand(*batch_shape, -1, -1)
    b = b.expand(*batch_shape, -1, -1)

    n = a.shape[-1]

    # Solve via Cholesky decomposition: B = L L^T
    # Transform to standard eigenvalue problem: L^{-1} A L^{-T} y = λ y
    # Then x = L^{-T} y
    try:
        L = torch.linalg.cholesky(b)
        # Solve L^{-1} A L^{-T} = C
        # First: Y = L^{-1} A (solve L Y = A)
        Y = torch.linalg.solve_triangular(L, a, upper=False)
        # Then: C = Y L^{-T} (solve C L^T = Y, i.e., L C^T = Y^T)
        C = torch.linalg.solve_triangular(L, Y.mH, upper=False).mH

        # Standard eigenvalue problem
        eigenvalues, Y = torch.linalg.eigh(C)

        # Back-transform eigenvectors: X = L^{-T} Y
        eigenvectors = torch.linalg.solve_triangular(L.mH, Y, upper=True)

        # Success
        info = torch.zeros(batch_shape, dtype=torch.int32, device=a.device)

    except RuntimeError:
        # Cholesky failed - B is not positive definite
        eigenvalues = torch.full(
            (*batch_shape, n), float("nan"), dtype=dtype, device=a.device
        )
        eigenvectors = torch.full(
            (*batch_shape, n, n), float("nan"), dtype=dtype, device=a.device
        )
        info = torch.ones(batch_shape, dtype=torch.int32, device=a.device)

    return SymmetricGeneralizedEigenvalueResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        info=info,
    )

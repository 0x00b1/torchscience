"""Symmetric generalized eigenvalue decomposition."""

import torch
from torch import Tensor
from torch.autograd import Function

from torchscience.linear_algebra.decomposition._result_types import (
    SymmetricGeneralizedEigenvalueResult,
)


class SymmetricGeneralizedEigenvalueFunction(Function):
    """Autograd function for symmetric generalized eigenvalue decomposition."""

    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor):
        # Promote to common dtype
        dtype = torch.promote_types(a.dtype, b.dtype)
        if dtype not in (torch.float32, torch.float64):
            dtype = torch.float64
        a = a.to(dtype)
        b = b.to(dtype)

        # Broadcast batch dimensions
        batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        a = a.expand(*batch_shape, -1, -1).contiguous()
        b = b.expand(*batch_shape, -1, -1).contiguous()

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
                (*batch_shape, n, n),
                float("nan"),
                dtype=dtype,
                device=a.device,
            )
            info = torch.ones(batch_shape, dtype=torch.int32, device=a.device)

        # Save for backward
        ctx.save_for_backward(eigenvalues, eigenvectors, a, b)

        return eigenvalues, eigenvectors, info

    @staticmethod
    def backward(ctx, grad_eigenvalues, grad_eigenvectors, grad_info):
        eigenvalues, eigenvectors, a, b = ctx.saved_tensors

        V = eigenvectors
        lam = eigenvalues

        # Gradient w.r.t. eigenvalues: dL/dA = V @ diag(dL/dlam) @ V^T
        # Gradient w.r.t. eigenvectors requires solving Sylvester equation

        grad_a = None
        grad_b = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            # For eigenvalue gradients
            if grad_eigenvalues is not None:
                # dL/dA = V @ diag(dL/dlam) @ V^T
                # V is B-orthonormal, so this gives the right gradient
                grad_a = V @ torch.diag_embed(grad_eigenvalues) @ V.mH

                # dL/dB = -V @ diag(lam * dL/dlam) @ V^T
                grad_b = -V @ torch.diag_embed(lam * grad_eigenvalues) @ V.mH

            if grad_eigenvectors is not None:
                # Full eigenvector gradient via perturbation theory
                # d(V_i) = sum_{j != i} V_j * (V_j^T B grad_V_i) / (lam_i - lam_j)

                # B-orthogonalize the gradient
                BV = b @ V

                # Compute the coefficient matrix
                # F_ij = 1 / (lam_i - lam_j) for i != j, 0 for i == j
                lam_diff = lam.unsqueeze(-1) - lam.unsqueeze(-2)
                # Avoid division by zero on diagonal
                F = torch.where(
                    lam_diff.abs() > 1e-10,
                    1.0 / lam_diff,
                    torch.zeros_like(lam_diff),
                )
                F.diagonal(dim1=-2, dim2=-1).zero_()

                # Project gradient onto eigenvector space
                proj = BV.mH @ grad_eigenvectors

                # Compute contribution to A and B gradients
                contrib = V @ (F * proj) @ V.mH

                if grad_a is None:
                    grad_a = b @ contrib @ b
                else:
                    grad_a = grad_a + b @ contrib @ b

                if grad_b is None:
                    grad_b = -a @ contrib @ b - b @ contrib @ a
                else:
                    grad_b = grad_b - a @ contrib @ b - b @ contrib @ a

        # Symmetrize gradients (A and B are symmetric)
        if grad_a is not None:
            grad_a = (grad_a + grad_a.mH) / 2
        if grad_b is not None:
            grad_b = (grad_b + grad_b.mH) / 2

        return grad_a, grad_b


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
        SymmetricGeneralizedEigenvalueFunction.apply(a, b)
    )

    return SymmetricGeneralizedEigenvalueResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        info=info,
    )

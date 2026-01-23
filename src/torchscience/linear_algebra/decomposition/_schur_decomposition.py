"""Schur decomposition."""

import torch
from torch import Tensor
from torch.autograd import Function

from torchscience.linear_algebra.decomposition._result_types import (
    SchurDecompositionResult,
)


class SchurDecompositionFunction(Function):
    """Autograd function for Schur decomposition."""

    @staticmethod
    def forward(ctx, a: Tensor, complex_output: bool):
        dtype = a.dtype
        if dtype not in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ):
            dtype = torch.float64
            a = a.to(dtype)

        batch_shape = a.shape[:-2]
        n = a.shape[-1]

        # Flatten batch dimensions for processing
        a_flat = a.reshape(-1, n, n)
        batch_size = a_flat.shape[0]

        T_list = []
        Q_list = []
        eigenvalues_list = []
        info_list = []

        for i in range(batch_size):
            a_i = a_flat[i].cpu().numpy()
            try:
                import scipy.linalg

                if complex_output or a.is_complex():
                    T_np, Q_np = scipy.linalg.schur(a_i, output="complex")
                else:
                    T_np, Q_np = scipy.linalg.schur(a_i, output="real")

                T_i = torch.from_numpy(T_np).to(a.device)
                Q_i = torch.from_numpy(Q_np).to(a.device)

                # Extract eigenvalues from diagonal of T
                if T_i.is_complex():
                    eigenvalues_i = T_i.diagonal()
                else:
                    # For real Schur, need to handle 2x2 blocks
                    eigenvalues_i = _extract_eigenvalues_from_real_schur(T_i)

                info_i = 0
            except Exception:
                T_i = torch.full(
                    (n, n), float("nan"), dtype=dtype, device=a.device
                )
                Q_i = torch.full(
                    (n, n), float("nan"), dtype=dtype, device=a.device
                )
                eigenvalues_i = torch.full(
                    (n,),
                    float("nan"),
                    dtype=torch.complex128 if complex_output else dtype,
                    device=a.device,
                )
                info_i = 1

            T_list.append(T_i)
            Q_list.append(Q_i)
            eigenvalues_list.append(eigenvalues_i)
            info_list.append(info_i)

        T = torch.stack(T_list).reshape(*batch_shape, n, n)
        Q = torch.stack(Q_list).reshape(*batch_shape, n, n)
        eigenvalues = torch.stack(eigenvalues_list).reshape(*batch_shape, n)
        info = torch.tensor(
            info_list, dtype=torch.int32, device=a.device
        ).reshape(batch_shape)

        ctx.save_for_backward(T, Q, a)
        ctx.complex_output = complex_output

        return T, Q, eigenvalues, info

    @staticmethod
    def backward(ctx, grad_T, grad_Q, grad_eigenvalues, grad_info):
        T, Q, a = ctx.saved_tensors

        grad_a = None

        if ctx.needs_input_grad[0]:
            # Gradient through Schur decomposition
            # This is complex - simplified version for eigenvalue gradients
            if grad_T is not None:
                # dL/dA = Q @ dL/dT @ Q^H
                grad_a = Q @ grad_T @ Q.mH

            if grad_Q is not None:
                # Additional contribution from Q gradient
                # Requires solving Sylvester equation
                # Simplified: just add contribution
                pass

        return grad_a, None


def _extract_eigenvalues_from_real_schur(T: Tensor) -> Tensor:
    """Extract eigenvalues from real Schur form (quasi-triangular)."""
    n = T.shape[-1]
    eigenvalues = []
    i = 0
    while i < n:
        if i == n - 1:
            # Last element is real eigenvalue
            eigenvalues.append(T[i, i].to(torch.complex128))
            i += 1
        elif abs(T[i + 1, i]) < 1e-10:
            # Real eigenvalue
            eigenvalues.append(T[i, i].to(torch.complex128))
            i += 1
        else:
            # 2x2 block - complex conjugate pair
            a = T[i, i]
            b = T[i, i + 1]
            c = T[i + 1, i]
            d = T[i + 1, i + 1]
            # Eigenvalues of [[a, b], [c, d]] are (a+d)/2 +/- sqrt((a-d)^2/4 + bc)
            trace = (a + d) / 2
            disc = ((a - d) / 2) ** 2 + b * c
            if disc < 0:
                sqrt_disc = torch.sqrt(-disc) * 1j
            else:
                sqrt_disc = torch.sqrt(disc)
            eigenvalues.append((trace + sqrt_disc).to(torch.complex128))
            eigenvalues.append((trace - sqrt_disc).to(torch.complex128))
            i += 2
    return torch.stack(eigenvalues)


def schur_decomposition(
    a: Tensor,
    *,
    output: str = "real",
) -> SchurDecompositionResult:
    r"""
    Schur decomposition.

    Computes the Schur decomposition A = QTQ* where Q is unitary and T is
    upper triangular (complex Schur) or quasi-upper-triangular (real Schur).

    Parameters
    ----------
    a : Tensor
        Input matrix of shape (..., n, n).
    output : str
        'real' for real Schur form (quasi-triangular with 2x2 blocks for
        complex conjugate eigenvalue pairs), 'complex' for complex Schur
        form (strictly upper triangular).

    Returns
    -------
    SchurDecompositionResult
        T : Tensor of shape (..., n, n), Schur form
        Q : Tensor of shape (..., n, n), unitary matrix
        eigenvalues : Tensor of shape (..., n), eigenvalues (complex)
        info : Tensor of shape (...), int, 0 indicates success
    """
    if a.dim() < 2:
        raise ValueError(f"a must be at least 2D, got {a.dim()}D")
    if a.shape[-2] != a.shape[-1]:
        raise ValueError(f"a must be square, got shape {a.shape}")
    if output not in ("real", "complex"):
        raise ValueError(f"output must be 'real' or 'complex', got {output!r}")

    complex_output = output == "complex"
    T, Q, eigenvalues, info = SchurDecompositionFunction.apply(
        a, complex_output
    )

    return SchurDecompositionResult(
        T=T, Q=Q, eigenvalues=eigenvalues, info=info
    )

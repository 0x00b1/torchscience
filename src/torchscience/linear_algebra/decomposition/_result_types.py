from typing import NamedTuple

from torch import Tensor


class GeneralizedEigenvalueResult(NamedTuple):
    """Result of generalized eigenvalue decomposition Ax = λBx."""

    eigenvalues: Tensor
    eigenvectors_left: Tensor
    eigenvectors_right: Tensor
    info: Tensor


class SymmetricGeneralizedEigenvalueResult(NamedTuple):
    """Result of symmetric generalized eigenvalue decomposition."""

    eigenvalues: Tensor
    eigenvectors: Tensor
    info: Tensor


class SchurDecompositionResult(NamedTuple):
    """Result of Schur decomposition A = QTQ*."""

    T: Tensor
    Q: Tensor
    eigenvalues: Tensor
    info: Tensor


class HessenbergResult(NamedTuple):
    """Result of Hessenberg decomposition A = QHQ*.

    H is upper Hessenberg (zeros below the first subdiagonal) and Q is unitary.
    """

    H: Tensor
    Q: Tensor
    info: Tensor


class GeneralizedSchurResult(NamedTuple):
    """Result of generalized Schur (QZ) decomposition.

    Factors matrix pencil (A, B) such that:
    - A = Q @ S @ Z.H
    - B = Q @ T @ Z.H

    where S and T are upper triangular (complex) or quasi-upper-triangular (real),
    and Q and Z are unitary matrices.

    The generalized eigenvalues are alpha[i] / beta[i].
    """

    S: Tensor  # (..., n, n) - Schur form of A
    T: Tensor  # (..., n, n) - Schur form of B
    alpha: Tensor  # (..., n) - complex, eigenvalue numerators
    beta: Tensor  # (..., n) - complex, eigenvalue denominators
    Q: Tensor  # (..., n, n) - Left unitary matrix
    Z: Tensor  # (..., n, n) - Right unitary matrix
    info: Tensor  # (...) - int, convergence info


class PolarDecompositionResult(NamedTuple):
    """Result of polar decomposition A = UP (right) or A = PU (left).

    Factors a matrix into the product of a unitary matrix U and a positive
    semidefinite Hermitian matrix P.

    For right polar (A = UP):
    - U has shape (..., m, n)
    - P has shape (..., n, n) and is positive semidefinite Hermitian

    For left polar (A = PU):
    - U has shape (..., m, n)
    - P has shape (..., m, m) and is positive semidefinite Hermitian
    """

    U: Tensor  # (..., m, n) - Unitary factor
    P: Tensor  # (..., n, n) or (..., m, m) - Positive semidefinite factor
    info: Tensor  # (...) - int, convergence info


class JordanDecompositionResult(NamedTuple):
    """Result of Jordan decomposition A = PJP^{-1}.

    Factors a square matrix into its Jordan normal form.

    For diagonalizable matrices, J is the diagonal matrix of eigenvalues.
    For defective matrices, J contains Jordan blocks with 1s on the superdiagonal.

    Note: This decomposition does not support gradients as the Jordan form
    is discontinuous with respect to matrix entries.
    """

    J: Tensor  # (..., n, n) - Jordan normal form (complex)
    P: Tensor  # (..., n, n) - Similarity transformation matrix (complex)
    info: (
        Tensor  # (...) - int, 0 indicates success, 1 indicates near-defective
    )

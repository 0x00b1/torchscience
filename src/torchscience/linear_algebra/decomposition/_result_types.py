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

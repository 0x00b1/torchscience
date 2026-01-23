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

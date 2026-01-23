"""Matrix decompositions with full PyTorch integration.

This module provides matrix decomposition operations with support for autograd,
torch.compile, autocast, vmap, complex tensors, and batching.

Functions
---------
generalized_eigenvalue
    Computes eigenvalues and eigenvectors for the generalized eigenvalue
    problem Ax = λBx where A and B are general square matrices.

symmetric_generalized_eigenvalue
    Computes eigenvalues and eigenvectors for the symmetric generalized
    eigenvalue problem Ax = λBx where A is symmetric and B is symmetric
    positive definite.

schur_decomposition
    Computes the Schur decomposition A = QTQ* where Q is unitary and T is
    upper triangular (complex) or quasi-upper-triangular (real).

generalized_schur
    Computes the generalized Schur (QZ) decomposition of matrix pencil (A, B)
    such that A = Q @ S @ Z.H and B = Q @ T @ Z.H.

hessenberg
    Computes the Hessenberg decomposition A = QHQ* where Q is unitary and H
    is upper Hessenberg (zeros below the first subdiagonal).

polar_decomposition
    Computes the polar decomposition A = UP (right) or A = PU (left) where
    U is unitary and P is positive semidefinite Hermitian.

jordan_decomposition
    Computes the Jordan decomposition A = PJP^{-1} where J is the Jordan
    normal form. Does not support gradients (discontinuous).

pivoted_lu
    Computes the pivoted LU decomposition PA = LU where P is a permutation
    matrix, L is unit lower triangular, and U is upper triangular.

pivoted_qr
    Computes the pivoted QR decomposition A[:, pivots] = QR where Q is
    orthogonal/unitary, R is upper triangular, and pivots are column
    permutation indices. Column pivoting provides numerical stability
    and rank-revealing properties.

rank_revealing_qr
    Computes the rank-revealing QR decomposition, which is pivoted QR with
    automatic numerical rank detection. The rank is determined by counting
    diagonal elements of R that exceed tol * |R[0,0]|.

ldl_decomposition
    Computes the LDL decomposition A = LDL* for symmetric/Hermitian matrices
    where L is unit lower triangular and D is diagonal.

Result Types
------------
GeneralizedEigenvalueResult
    Named tuple with eigenvalues, eigenvectors_left, eigenvectors_right, info.

SymmetricGeneralizedEigenvalueResult
    Named tuple with eigenvalues, eigenvectors, info.

SchurDecompositionResult
    Named tuple with T, Q, eigenvalues, info.

GeneralizedSchurResult
    Named tuple with S, T, alpha, beta, Q, Z, info.

HessenbergResult
    Named tuple with H, Q, info.

PolarDecompositionResult
    Named tuple with U, P, info.

JordanDecompositionResult
    Named tuple with J, P, info.

PivotedLUResult
    Named tuple with L, U, pivots, info.

PivotedQRResult
    Named tuple with Q, R, pivots, info.

RankRevealingQRResult
    Named tuple with Q, R, pivots, rank, info.

LDLDecompositionResult
    Named tuple with L, D, pivots, info.
"""

from torchscience.linear_algebra.decomposition._generalized_eigenvalue import (
    generalized_eigenvalue,
)
from torchscience.linear_algebra.decomposition._generalized_schur import (
    generalized_schur,
)
from torchscience.linear_algebra.decomposition._hessenberg import (
    hessenberg,
)
from torchscience.linear_algebra.decomposition._jordan_decomposition import (
    jordan_decomposition,
)
from torchscience.linear_algebra.decomposition._ldl_decomposition import (
    ldl_decomposition,
)
from torchscience.linear_algebra.decomposition._pivoted_lu import (
    pivoted_lu,
)
from torchscience.linear_algebra.decomposition._pivoted_qr import (
    pivoted_qr,
)
from torchscience.linear_algebra.decomposition._polar_decomposition import (
    polar_decomposition,
)
from torchscience.linear_algebra.decomposition._rank_revealing_qr import (
    rank_revealing_qr,
)
from torchscience.linear_algebra.decomposition._result_types import (
    GeneralizedEigenvalueResult,
    GeneralizedSchurResult,
    HessenbergResult,
    JordanDecompositionResult,
    LDLDecompositionResult,
    PivotedLUResult,
    PivotedQRResult,
    PolarDecompositionResult,
    RankRevealingQRResult,
    SchurDecompositionResult,
    SymmetricGeneralizedEigenvalueResult,
)
from torchscience.linear_algebra.decomposition._schur_decomposition import (
    schur_decomposition,
)
from torchscience.linear_algebra.decomposition._symmetric_generalized_eigenvalue import (
    symmetric_generalized_eigenvalue,
)

__all__ = [
    "GeneralizedEigenvalueResult",
    "GeneralizedSchurResult",
    "HessenbergResult",
    "JordanDecompositionResult",
    "LDLDecompositionResult",
    "PivotedLUResult",
    "PivotedQRResult",
    "PolarDecompositionResult",
    "RankRevealingQRResult",
    "SymmetricGeneralizedEigenvalueResult",
    "SchurDecompositionResult",
    "generalized_eigenvalue",
    "generalized_schur",
    "hessenberg",
    "jordan_decomposition",
    "ldl_decomposition",
    "pivoted_lu",
    "pivoted_qr",
    "polar_decomposition",
    "rank_revealing_qr",
    "schur_decomposition",
    "symmetric_generalized_eigenvalue",
]

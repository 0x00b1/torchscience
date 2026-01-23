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

hessenberg
    Computes the Hessenberg decomposition A = QHQ* where Q is unitary and H
    is upper Hessenberg (zeros below the first subdiagonal).

Result Types
------------
GeneralizedEigenvalueResult
    Named tuple with eigenvalues, eigenvectors_left, eigenvectors_right, info.

SymmetricGeneralizedEigenvalueResult
    Named tuple with eigenvalues, eigenvectors, info.

SchurDecompositionResult
    Named tuple with T, Q, eigenvalues, info.

HessenbergResult
    Named tuple with H, Q, info.
"""

from torchscience.linear_algebra.decomposition._generalized_eigenvalue import (
    generalized_eigenvalue,
)
from torchscience.linear_algebra.decomposition._hessenberg import (
    hessenberg,
)
from torchscience.linear_algebra.decomposition._result_types import (
    GeneralizedEigenvalueResult,
    HessenbergResult,
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
    "HessenbergResult",
    "SymmetricGeneralizedEigenvalueResult",
    "SchurDecompositionResult",
    "generalized_eigenvalue",
    "hessenberg",
    "schur_decomposition",
    "symmetric_generalized_eigenvalue",
]

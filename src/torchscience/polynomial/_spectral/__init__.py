"""Spectral methods for polynomial-based differentiation and integration.

This module provides differentiation matrices and collocation points
for spectral methods commonly used in solving differential equations
and high-accuracy numerical computation.

Differentiation Matrices
------------------------
chebyshev_differentiation_matrix
    First-order Chebyshev differentiation matrix on [-1, 1].
chebyshev_differentiation_matrix_2
    Second-order Chebyshev differentiation matrix.
chebyshev_differentiation_matrix_scaled
    Chebyshev differentiation matrix for arbitrary interval [a, b].
lagrange_differentiation_matrix
    Differentiation matrix for arbitrary collocation points.
legendre_differentiation_matrix
    Differentiation matrix for Legendre-Gauss-Lobatto points.

Collocation Points
------------------
chebyshev_points
    Chebyshev-Gauss-Lobatto points (extrema of T_n).
legendre_gauss_lobatto_points
    Legendre-Gauss-Lobatto points (roots of (1-x^2)P'_n).
legendre_gauss_points
    Legendre-Gauss points (roots of P_n).
uniform_points
    Uniformly spaced points.

Integration
-----------
integration_matrix
    Spectral integration matrix for arbitrary points.

Examples
--------
>>> import torch
>>> from torchscience.polynomial._spectral import (
...     chebyshev_differentiation_matrix,
...     chebyshev_points,
... )

>>> # Solve u'' = -u with u(1) = 0, u(-1) = 0
>>> n = 16
>>> D = chebyshev_differentiation_matrix(n)
>>> D2 = D @ D  # Second derivative
>>> x = chebyshev_points(n)

>>> # Apply boundary conditions by removing first and last rows/cols
>>> A = D2[1:-1, 1:-1] + torch.eye(n-1)
>>> # Solve for interior points...

References
----------
.. [1] Trefethen, L. N. (2000). Spectral Methods in MATLAB. SIAM.
.. [2] Fornberg, B. (1998). A Practical Guide to Pseudospectral Methods.
       Cambridge University Press.
"""

from ._chebyshev_differentiation_matrix import (
    chebyshev_differentiation_matrix,
    chebyshev_differentiation_matrix_2,
    chebyshev_differentiation_matrix_scaled,
    chebyshev_points,
)
from ._collocation_points import (
    legendre_gauss_lobatto_points,
    legendre_gauss_points,
    uniform_points,
)
from ._differentiation_matrix import (
    integration_matrix,
    lagrange_differentiation_matrix,
    legendre_differentiation_matrix,
)

__all__ = [
    # Chebyshev
    "chebyshev_differentiation_matrix",
    "chebyshev_differentiation_matrix_2",
    "chebyshev_differentiation_matrix_scaled",
    "chebyshev_points",
    # Legendre
    "legendre_differentiation_matrix",
    "legendre_gauss_lobatto_points",
    "legendre_gauss_points",
    # General
    "integration_matrix",
    "lagrange_differentiation_matrix",
    "uniform_points",
]

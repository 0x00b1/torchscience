"""Finite element method module.

This module provides tools for finite element computations including:
    - Quadrature rules for reference elements
    - Basis functions (Lagrange, etc.)
    - DOF maps and assembly utilities
    - Local element matrices
"""

from torchscience.finite_element_method._assembly import (
    assemble_matrix,
    assemble_vector,
)
from torchscience.finite_element_method._basis import (
    lagrange_basis,
    lagrange_basis_gradient,
)
from torchscience.finite_element_method._boundary import (
    apply_dirichlet_elimination,
    apply_dirichlet_penalty,
    boundary_dofs,
)
from torchscience.finite_element_method._dof_map import DOFMap, dof_map
from torchscience.finite_element_method._local_matrices import (
    local_mass_matrices,
    local_stiffness_matrices,
)
from torchscience.finite_element_method._quadrature import quadrature_points

__all__ = [
    "DOFMap",
    "apply_dirichlet_elimination",
    "apply_dirichlet_penalty",
    "assemble_matrix",
    "assemble_vector",
    "boundary_dofs",
    "dof_map",
    "lagrange_basis",
    "lagrange_basis_gradient",
    "local_mass_matrices",
    "local_stiffness_matrices",
    "quadrature_points",
]

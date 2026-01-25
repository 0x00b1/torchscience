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
    apply_neumann,
    boundary_dofs,
)
from torchscience.finite_element_method._dof_map import DOFMap, dof_map
from torchscience.finite_element_method._local_matrices import (
    local_mass_matrices,
    local_stiffness_matrices,
)
from torchscience.finite_element_method._physics import solve_poisson
from torchscience.finite_element_method._quadrature import quadrature_points
from torchscience.finite_element_method._solve import solve_cg, solve_direct
from torchscience.finite_element_method._weak_form import (
    BasisValues,
    WeakForm,
    assemble_weak_form,
    mass_form,
    poisson_form,
)

__all__ = [
    "BasisValues",
    "DOFMap",
    "WeakForm",
    "assemble_weak_form",
    "apply_dirichlet_elimination",
    "apply_dirichlet_penalty",
    "apply_neumann",
    "assemble_matrix",
    "assemble_vector",
    "boundary_dofs",
    "dof_map",
    "lagrange_basis",
    "lagrange_basis_gradient",
    "local_mass_matrices",
    "local_stiffness_matrices",
    "mass_form",
    "poisson_form",
    "quadrature_points",
    "solve_cg",
    "solve_direct",
    "solve_poisson",
]

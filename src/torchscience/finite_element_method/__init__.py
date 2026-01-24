"""Finite element method module.

This module provides tools for finite element computations including:
    - Quadrature rules for reference elements
    - Basis functions (Lagrange, etc.)
    - DOF maps and assembly utilities
"""

from torchscience.finite_element_method._basis import (
    lagrange_basis,
    lagrange_basis_gradient,
)
from torchscience.finite_element_method._dof_map import DOFMap, dof_map
from torchscience.finite_element_method._quadrature import quadrature_points

__all__ = [
    "DOFMap",
    "dof_map",
    "lagrange_basis",
    "lagrange_basis_gradient",
    "quadrature_points",
]

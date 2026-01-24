"""Finite element method module.

This module provides tools for finite element computations including:
    - Quadrature rules for reference elements
    - Basis functions (Lagrange, etc.)
    - DOF maps and assembly utilities
"""

from torchscience.finite_element_method._quadrature import quadrature_points

__all__ = [
    "quadrature_points",
]

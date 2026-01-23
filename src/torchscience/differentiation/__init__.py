"""Differentiation module: finite difference operators for numerical differentiation."""

from torchscience.differentiation._apply import apply_stencil
from torchscience.differentiation._biharmonic import biharmonic
from torchscience.differentiation._biharmonic_stencil import biharmonic_stencil
from torchscience.differentiation._curl import curl
from torchscience.differentiation._derivative import derivative
from torchscience.differentiation._divergence import divergence
from torchscience.differentiation._exceptions import (
    BoundaryError,
    DifferentiationError,
    StencilError,
)
from torchscience.differentiation._finite_difference_stencil import (
    finite_difference_stencil,
)
from torchscience.differentiation._fornberg_weights import fornberg_weights
from torchscience.differentiation._gradient import gradient
from torchscience.differentiation._gradient_stencils import gradient_stencils
from torchscience.differentiation._grid import IrregularMesh, RegularGrid
from torchscience.differentiation._hessian import hessian
from torchscience.differentiation._jacobian import jacobian
from torchscience.differentiation._laplacian import laplacian
from torchscience.differentiation._laplacian_stencil import laplacian_stencil
from torchscience.differentiation._path import Path, Surface
from torchscience.differentiation._richardson import richardson_extrapolation
from torchscience.differentiation._spectral_derivative import (
    spectral_derivative,
)
from torchscience.differentiation._spectral_gradient import (
    spectral_gradient,
)
from torchscience.differentiation._spectral_laplacian import (
    spectral_laplacian,
)
from torchscience.differentiation._stencil import FiniteDifferenceStencil

__all__ = [
    "BoundaryError",
    "DifferentiationError",
    "FiniteDifferenceStencil",
    "IrregularMesh",
    "Path",
    "RegularGrid",
    "StencilError",
    "Surface",
    "apply_stencil",
    "biharmonic",
    "biharmonic_stencil",
    "curl",
    "derivative",
    "divergence",
    "finite_difference_stencil",
    "fornberg_weights",
    "gradient",
    "gradient_stencils",
    "hessian",
    "jacobian",
    "laplacian",
    "laplacian_stencil",
    "richardson_extrapolation",
    "spectral_derivative",
    "spectral_gradient",
    "spectral_laplacian",
]

"""Differentiation module: finite difference operators for numerical differentiation."""

from torchscience.differentiation._advect import advect
from torchscience.differentiation._apply import apply_stencil
from torchscience.differentiation._biharmonic import biharmonic
from torchscience.differentiation._biharmonic_stencil import biharmonic_stencil
from torchscience.differentiation._chebyshev_derivative import (
    chebyshev_derivative,
    chebyshev_points,
)
from torchscience.differentiation._curl import curl
from torchscience.differentiation._derivative import derivative
from torchscience.differentiation._diffuse import diffuse
from torchscience.differentiation._divergence import divergence
from torchscience.differentiation._enstrophy import enstrophy
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
from torchscience.differentiation._helicity import helicity
from torchscience.differentiation._hessian import hessian
from torchscience.differentiation._jacobian import jacobian
from torchscience.differentiation._laplacian import laplacian
from torchscience.differentiation._laplacian_stencil import laplacian_stencil
from torchscience.differentiation._line_integral import (
    circulation,
    line_integral,
)
from torchscience.differentiation._material_derivative import (
    material_derivative,
)
from torchscience.differentiation._path import Path, Surface
from torchscience.differentiation._q_criterion import q_criterion
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
from torchscience.differentiation._strain_tensor import strain_tensor
from torchscience.differentiation._vorticity import vorticity

__all__ = [
    "BoundaryError",
    "DifferentiationError",
    "FiniteDifferenceStencil",
    "IrregularMesh",
    "Path",
    "RegularGrid",
    "StencilError",
    "Surface",
    "advect",
    "apply_stencil",
    "biharmonic",
    "biharmonic_stencil",
    "chebyshev_derivative",
    "chebyshev_points",
    "circulation",
    "curl",
    "derivative",
    "diffuse",
    "divergence",
    "enstrophy",
    "finite_difference_stencil",
    "fornberg_weights",
    "gradient",
    "gradient_stencils",
    "helicity",
    "hessian",
    "jacobian",
    "laplacian",
    "laplacian_stencil",
    "line_integral",
    "material_derivative",
    "q_criterion",
    "richardson_extrapolation",
    "spectral_derivative",
    "spectral_gradient",
    "spectral_laplacian",
    "strain_tensor",
    "vorticity",
]

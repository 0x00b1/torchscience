"""
Numerical integration (quadrature) module.

Sample-based integration (operates on pre-computed values):
    trapezoid, cumulative_trapezoid, simpson, cumulative_simpson

Function-based integration (evaluates callable):
    fixed_quad, quad, quad_info, dblquad

Quadrature rule classes:
    GaussLegendre, GaussKronrod

Node/weight computation for Gaussian quadrature:
    gauss_legendre_nodes_weights, gauss_hermite_nodes_weights,
    gauss_laguerre_nodes_weights, gauss_chebyshev_nodes_weights,
    gauss_jacobi_nodes_weights

Exceptions:
    QuadratureWarning, IntegrationError
"""

from torchscience.integration.quadrature._dblquad import dblquad
from torchscience.integration.quadrature._exceptions import (
    IntegrationError,
    QuadratureWarning,
)
from torchscience.integration.quadrature._fixed_quad import fixed_quad
from torchscience.integration.quadrature._nodes import (
    gauss_chebyshev_nodes_weights,
    gauss_hermite_nodes_weights,
    gauss_jacobi_nodes_weights,
    gauss_kronrod_nodes_weights,
    gauss_laguerre_nodes_weights,
    gauss_legendre_nodes_weights,
)
from torchscience.integration.quadrature._quad import quad, quad_info
from torchscience.integration.quadrature._rules import (
    GaussKronrod,
    GaussLegendre,
)
from torchscience.integration.quadrature._simpson import (
    cumulative_simpson,
    simpson,
)
from torchscience.integration.quadrature._trapezoid import (
    cumulative_trapezoid,
    trapezoid,
)

__all__ = [
    # Sample-based
    "trapezoid",
    "cumulative_trapezoid",
    "simpson",
    "cumulative_simpson",
    # Function-based
    "fixed_quad",
    "quad",
    "quad_info",
    "dblquad",
    # Rule classes
    "GaussLegendre",
    "GaussKronrod",
    # Node/weight computation
    "gauss_legendre_nodes_weights",
    "gauss_hermite_nodes_weights",
    "gauss_laguerre_nodes_weights",
    "gauss_chebyshev_nodes_weights",
    "gauss_jacobi_nodes_weights",
    "gauss_kronrod_nodes_weights",
    # Exceptions
    "QuadratureWarning",
    "IntegrationError",
]

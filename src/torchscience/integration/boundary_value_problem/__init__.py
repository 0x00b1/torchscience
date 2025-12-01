"""Boundary value problem solvers.

This module provides differentiable solvers for two-point boundary value
problems (BVPs) for systems of ordinary differential equations.

Main API
--------
solve_bvp : Solve BVP using 4th-order Lobatto collocation

Data Structures
---------------
BVPSolution : Solution container with interpolation support

Exceptions
----------
BVPError : Base exception for BVP errors
BVPConvergenceError : Newton iteration failed
BVPMeshError : Mesh refinement exceeded limits
BVPSingularJacobianError : Jacobian is singular

Note: Exception names use BVP prefix to avoid collision with IVP exceptions.
"""

from torchscience.integration.boundary_value_problem._adjoint import (
    bvp_adjoint,
)
from torchscience.integration.boundary_value_problem._exceptions import (
    BVPConvergenceError,
    BVPError,
    BVPMeshError,
    BVPSingularJacobianError,
)
from torchscience.integration.boundary_value_problem._solution import (
    BVPSolution,
)
from torchscience.integration.boundary_value_problem._solve_bvp import (
    solve_bvp,
)

__all__ = [
    # Exceptions
    "BVPError",
    "BVPConvergenceError",
    "BVPMeshError",
    "BVPSingularJacobianError",
    # Solution
    "BVPSolution",
    # Solvers
    "solve_bvp",
    # Wrappers (matches IVP's adjoint pattern)
    "bvp_adjoint",
]

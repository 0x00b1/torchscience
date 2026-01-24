"""
Numerical integration methods.

This module provides differentiable solvers for:
- Initial value problems (ODEs)
- Boundary value problems (BVPs)

Initial Value Problem Solvers
-----------------------------
euler
    Forward Euler method (1st order, fixed step, explicit).
midpoint
    Explicit midpoint method (2nd order, fixed step).
runge_kutta_4
    Classic 4th-order Runge-Kutta (fixed step, explicit).
dormand_prince_5
    Dormand-Prince 5(4) adaptive method (explicit).
backward_euler
    Backward Euler method (1st order, fixed step, implicit).
bdf
    Variable-order BDF method (orders 1-5, stiff).
radau
    Radau IIA method (3-stage, order 5, L-stable).
stormer_verlet
    Stormer-Verlet symplectic integrator.
yoshida4
    Yoshida 4th-order symplectic integrator.
implicit_midpoint
    Implicit midpoint method (symplectic and A-stable).
adjoint
    Wrapper for continuous adjoint method.
solve_ivp
    Unified IVP solver API.

Boundary Value Problem Solvers
------------------------------
solve_bvp
    Solve BVP using 4th-order Lobatto collocation.
bvp_adjoint
    Wrapper for BVP adjoint method.
BVPSolution
    Solution container for BVPs.
"""

# BVP exports
# IVP exports
from torchscience.ordinary_differential_equation._adams import adams
from torchscience.ordinary_differential_equation._asynchronous_leapfrog import (
    asynchronous_leapfrog,
)
from torchscience.ordinary_differential_equation._backward_euler import (
    backward_euler,
)
from torchscience.ordinary_differential_equation._batched import (
    solve_ivp_batched,
)
from torchscience.ordinary_differential_equation._bdf import bdf
from torchscience.ordinary_differential_equation._bvp_adjoint import (
    bvp_adjoint,
)
from torchscience.ordinary_differential_equation._bvp_exceptions import (
    BVPConvergenceError,
    BVPError,
    BVPMeshError,
    BVPSingularJacobianError,
)
from torchscience.ordinary_differential_equation._bvp_solution import (
    BVPSolution,
)
from torchscience.ordinary_differential_equation._cnf import (
    cnf_dynamics,
    exact_trace,
    hutchinson_trace,
)
from torchscience.ordinary_differential_equation._compile_utils import (
    compile_solver,
    is_compile_compatible,
)
from torchscience.ordinary_differential_equation._dop853 import dop853
from torchscience.ordinary_differential_equation._dormand_prince_5 import (
    dormand_prince_5,
)
from torchscience.ordinary_differential_equation._euler import euler
from torchscience.ordinary_differential_equation._implicit_midpoint import (
    implicit_midpoint,
)
from torchscience.ordinary_differential_equation._ivp_adjoint import adjoint
from torchscience.ordinary_differential_equation._ivp_exceptions import (
    ConvergenceError,
    MaxStepsExceeded,
    ODESolverError,
    StepSizeError,
)
from torchscience.ordinary_differential_equation._midpoint import midpoint
from torchscience.ordinary_differential_equation._radau import radau
from torchscience.ordinary_differential_equation._recommend import (
    analyze_problem,
    recommend_solver,
)
from torchscience.ordinary_differential_equation._reversible_heun import (
    reversible_heun,
)
from torchscience.ordinary_differential_equation._runge_kutta_4 import (
    runge_kutta_4,
)
from torchscience.ordinary_differential_equation._second_order import (
    solve_ivp_hvp,
)
from torchscience.ordinary_differential_equation._sensitivity import (
    solve_ivp_sensitivity,
)
from torchscience.ordinary_differential_equation._solve_bvp import solve_bvp
from torchscience.ordinary_differential_equation._solve_ivp import (
    ODESolution,
    solve_ivp,
)
from torchscience.ordinary_differential_equation._stormer_verlet import (
    stormer_verlet,
)
from torchscience.ordinary_differential_equation._yoshida import yoshida4

__all__ = [
    # BVP Exceptions
    "BVPError",
    "BVPConvergenceError",
    "BVPMeshError",
    "BVPSingularJacobianError",
    # BVP Solution
    "BVPSolution",
    # BVP Solvers
    "solve_bvp",
    "bvp_adjoint",
    # IVP Exceptions
    "ConvergenceError",
    "MaxStepsExceeded",
    "ODESolverError",
    "StepSizeError",
    # IVP Explicit solvers
    "euler",
    "midpoint",
    "runge_kutta_4",
    "dormand_prince_5",
    # IVP Implicit solvers
    "backward_euler",
    # IVP Stiff solvers
    "bdf",
    "radau",
    # Symplectic integrators
    "stormer_verlet",
    "yoshida4",
    "implicit_midpoint",
    # Wrappers
    "adjoint",
    # Unified API
    "ODESolution",
    "solve_ivp",
    "solve_ivp_batched",
    "solve_ivp_hvp",
    "solve_ivp_sensitivity",
    # CNF utilities
    "cnf_dynamics",
    "exact_trace",
    "hutchinson_trace",
    # Neural ODE optimizations
    "reversible_heun",
    "asynchronous_leapfrog",
    "compile_solver",
    "is_compile_compatible",
    # High-order adaptive
    "dop853",
    # Multistep methods
    "adams",
    # Utilities
    "recommend_solver",
    "analyze_problem",
]

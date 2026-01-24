"""
Numerical integration methods.

Submodules
----------
boundary_value_problem
    Solvers for boundary value problems (BVPs).
initial_value_problem
    Solvers for initial value problems (ODEs).
"""

from torchscience.integration import (
    boundary_value_problem,
    initial_value_problem,
)

__all__ = [
    "boundary_value_problem",
    "initial_value_problem",
]

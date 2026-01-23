from ._brent import brent
from ._convergence import check_convergence, default_tolerances
from ._differentiation import (
    compute_derivative,
    compute_jacobian,
    compute_second_derivative,
)
from ._exceptions import BracketError, DerivativeError, RootFindingError
from ._fixed_point import fixed_point
from ._halley import halley
from ._newton import newton
from ._ridder import ridder
from ._secant import secant

__all__ = [
    "brent",
    "check_convergence",
    "compute_derivative",
    "compute_jacobian",
    "compute_second_derivative",
    "default_tolerances",
    "fixed_point",
    "halley",
    "newton",
    "ridder",
    "secant",
    "BracketError",
    "DerivativeError",
    "RootFindingError",
]

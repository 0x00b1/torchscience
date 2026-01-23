from ._brent import brent
from ._convergence import check_convergence, default_tolerances
from ._differentiation import (
    compute_derivative,
    compute_jacobian,
    compute_second_derivative,
)
from ._exceptions import BracketError, DerivativeError, RootFindingError
from ._newton import newton
from ._secant import secant

__all__ = [
    "brent",
    "check_convergence",
    "compute_derivative",
    "compute_jacobian",
    "compute_second_derivative",
    "default_tolerances",
    "newton",
    "secant",
    "BracketError",
    "DerivativeError",
    "RootFindingError",
]

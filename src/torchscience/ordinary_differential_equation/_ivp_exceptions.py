"""Exceptions for ODE solvers."""

from torchscience.ordinary_differential_equation._exceptions import (
    IntegrationError,
)


class ODESolverError(IntegrationError):
    """Base exception for ODE solver errors.

    Inherits from IntegrationError for consistency with BVP module.
    """

    pass


class MaxStepsExceeded(ODESolverError):
    """Raised when adaptive solver exceeds max_steps."""

    pass


class StepSizeError(ODESolverError):
    """Raised when adaptive step size falls below dt_min."""

    pass


class ConvergenceError(ODESolverError):
    """Raised when implicit solver Newton iteration fails to converge."""

    pass

"""Exceptions for boundary value problem solvers."""

from torchscience.integration._exceptions import IntegrationError


class BVPError(IntegrationError):
    """Base exception for BVP solver errors.

    Inherits from IntegrationError for consistency with IVP module.
    """

    pass


class BVPSingularJacobianError(BVPError):
    """Raised when Jacobian is singular during Newton iteration."""

    def __init__(self, iteration: int, message: str = ""):
        self.iteration = iteration
        super().__init__(
            f"Singular Jacobian at iteration {iteration}. {message}"
        )


class BVPConvergenceError(BVPError):
    """Raised when Newton iteration fails to converge.

    Note: Named BVPConvergenceError to distinguish from IVP's ConvergenceError.
    """

    def __init__(self, iteration: int, residual: float, message: str = ""):
        self.iteration = iteration
        self.residual = residual
        super().__init__(
            f"Newton failed at iteration {iteration} with residual {residual:.2e}. {message}"
        )


class BVPMeshError(BVPError):
    """Raised when mesh refinement exceeds max_nodes."""

    def __init__(self, current_nodes: int, max_nodes: int):
        self.current_nodes = current_nodes
        self.max_nodes = max_nodes
        super().__init__(
            f"Mesh refinement exceeded max_nodes={max_nodes}. Current: {current_nodes}"
        )

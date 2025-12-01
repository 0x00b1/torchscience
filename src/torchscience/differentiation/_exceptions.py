"""Differentiation module exceptions."""


class DifferentiationError(Exception):
    """Base exception for differentiation operations."""

    pass


class StencilError(DifferentiationError):
    """Error in stencil construction or application."""

    pass


class BoundaryError(DifferentiationError):
    """Error in boundary handling."""

    pass

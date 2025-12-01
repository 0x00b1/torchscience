"""Geometry module exceptions."""


class GeometryError(Exception):
    """Base exception for geometry operations."""

    pass


class DegenerateInputError(GeometryError):
    """Input is degenerate (e.g., collinear points in 2D)."""

    pass


class InsufficientPointsError(GeometryError):
    """Not enough points for the requested operation."""

    pass

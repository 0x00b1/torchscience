"""Exception classes for root finding module."""


class RootFindingError(Exception):
    """Base exception for root finding errors."""

    pass


class BracketError(RootFindingError):
    """Raised when bracket doesn't contain sign change."""

    pass


class DerivativeError(RootFindingError):
    """Raised when derivative computation fails (e.g., zero derivative)."""

    pass

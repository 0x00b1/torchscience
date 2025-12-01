"""Probability module exceptions."""

__all__ = ["ProbabilityError", "DomainError"]


class ProbabilityError(ValueError):
    """Base exception for probability module errors."""

    pass


class DomainError(ProbabilityError):
    """Raised when input is outside the valid domain."""

    pass

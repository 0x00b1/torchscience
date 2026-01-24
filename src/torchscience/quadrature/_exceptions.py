"""Exceptions for quadrature integration."""


class QuadratureWarning(UserWarning):
    """Warning for quadrature issues (e.g., slow convergence)."""

    pass


class IntegrationError(Exception):
    """Error when integration fails to converge."""

    pass

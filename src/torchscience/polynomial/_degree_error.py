from torchscience.polynomial._polynomial_error import PolynomialError


class DegreeError(PolynomialError):
    """Raised when degree is invalid for operation."""

    pass

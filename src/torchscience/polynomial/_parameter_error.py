from ._polynomial_error import PolynomialError


class ParameterError(PolynomialError):
    """Invalid polynomial parameters.

    Raised when parameterized polynomials (Jacobi, Gegenbauer, etc.)
    have invalid parameters (e.g., alpha <= -1 for Jacobi polynomials).
    """

    pass

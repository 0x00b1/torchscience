"""Exception hierarchy for polynomial operations."""

from torchscience.polynomial._polynomial_error import PolynomialError


class DomainError(PolynomialError):
    """Operation outside valid domain.

    Raised when fitting points or evaluating polynomials outside
    their natural domain (e.g., Chebyshev polynomials outside [-1, 1]).
    """

    pass


class ParameterError(PolynomialError):
    """Invalid polynomial parameters.

    Raised when parameterized polynomials (Jacobi, Gegenbauer, etc.)
    have invalid parameters (e.g., alpha <= -1 for Jacobi polynomials).
    """

    pass


class ParameterMismatchError(PolynomialError):
    """Arithmetic between polynomials with different parameters.

    Raised when arithmetic operations are attempted between
    parameterized polynomials with incompatible parameters
    (e.g., adding Jacobi polynomials with different alpha values).
    """

    pass

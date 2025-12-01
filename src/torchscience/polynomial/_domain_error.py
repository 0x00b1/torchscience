from ._polynomial_error import PolynomialError


class DomainError(PolynomialError):
    """Operation outside valid domain.

    Raised when fitting points or evaluating polynomials outside
    their natural domain (e.g., Chebyshev polynomials outside [-1, 1]).
    """

    pass

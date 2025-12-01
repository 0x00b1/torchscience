from ._polynomial_error import PolynomialError


class ParameterMismatchError(PolynomialError):
    """Arithmetic between polynomials with different parameters.

    Raised when arithmetic operations are attempted between
    parameterized polynomials with incompatible parameters
    (e.g., adding Jacobi polynomials with different alpha values).
    """

    pass

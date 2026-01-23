"""Create Euler polynomial series from roots."""

from torch import Tensor

from ._euler_polynomial_e import (
    EulerPolynomialE,
)
from ._polynomial_to_euler_polynomial_e import (
    polynomial_to_euler_polynomial_e,
)


def euler_polynomial_e_from_roots(roots: Tensor) -> EulerPolynomialE:
    """Create Euler polynomial series from its roots.

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial.

    Returns
    -------
    EulerPolynomialE
        Euler polynomial series with the given roots.
    """
    from torchscience.polynomial._polynomial._polynomial_from_roots import (
        polynomial_from_roots,
    )

    # Create standard polynomial from roots
    p = polynomial_from_roots(roots)

    # Convert to Euler basis
    return polynomial_to_euler_polynomial_e(p)

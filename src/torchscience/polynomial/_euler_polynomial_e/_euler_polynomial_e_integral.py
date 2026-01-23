"""Definite integral of Euler polynomial series."""

from torch import Tensor

from ._euler_polynomial_e import EulerPolynomialE
from ._euler_polynomial_e_antiderivative import (
    euler_polynomial_e_antiderivative,
)
from ._euler_polynomial_e_evaluate import euler_polynomial_e_evaluate


def euler_polynomial_e_integral(
    a: EulerPolynomialE,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Compute definite integral of Euler polynomial series.

    Parameters
    ----------
    a : EulerPolynomialE
        Series to integrate.
    lower : Tensor
        Lower bound of integration.
    upper : Tensor
        Upper bound of integration.

    Returns
    -------
    Tensor
        Definite integral values.
    """
    # Get antiderivative with constant 0
    A = euler_polynomial_e_antiderivative(a, order=1, constant=0.0)

    # Evaluate at bounds
    upper_val = euler_polynomial_e_evaluate(A, upper)
    lower_val = euler_polynomial_e_evaluate(A, lower)

    return upper_val - lower_val

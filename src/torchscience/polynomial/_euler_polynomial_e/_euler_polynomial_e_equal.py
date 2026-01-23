"""Equality comparison for Euler polynomial series."""

import torch

from ._euler_polynomial_e import EulerPolynomialE
from ._euler_polynomial_e_trim import euler_polynomial_e_trim


def euler_polynomial_e_equal(
    a: EulerPolynomialE,
    b: EulerPolynomialE,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Check if two Euler polynomial series are equal.

    Parameters
    ----------
    a : EulerPolynomialE
        First series.
    b : EulerPolynomialE
        Second series.
    rtol : float, optional
        Relative tolerance. Default is 1e-5.
    atol : float, optional
        Absolute tolerance. Default is 1e-8.

    Returns
    -------
    bool
        True if series are approximately equal.
    """
    # Trim both series
    a_trimmed = euler_polynomial_e_trim(a, tol=atol)
    b_trimmed = euler_polynomial_e_trim(b, tol=atol)

    a_coeffs = a_trimmed.as_subclass(torch.Tensor)
    b_coeffs = b_trimmed.as_subclass(torch.Tensor)

    # Must have same shape after trimming
    if a_coeffs.shape != b_coeffs.shape:
        return False

    return torch.allclose(a_coeffs, b_coeffs, rtol=rtol, atol=atol)

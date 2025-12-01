from typing import Tuple

import torch

from .._parameter_mismatch_error import ParameterMismatchError
from .._polynomial import polynomial_divmod
from ._jacobi_polynomial_p import JacobiPolynomialP
from ._jacobi_polynomial_p_to_polynomial import (
    jacobi_polynomial_p_to_polynomial,
)
from ._polynomial_to_jacobi_polynomial_p import (
    polynomial_to_jacobi_polynomial_p,
)


def jacobi_polynomial_p_divmod(
    a: JacobiPolynomialP,
    b: JacobiPolynomialP,
) -> Tuple[JacobiPolynomialP, JacobiPolynomialP]:
    """Divide two Jacobi series with remainder.

    Returns quotient q and remainder r such that a = b*q + r.

    Parameters
    ----------
    a : JacobiPolynomialP
        Dividend.
    b : JacobiPolynomialP
        Divisor.

    Returns
    -------
    Tuple[JacobiPolynomialP, JacobiPolynomialP]
        (quotient, remainder)

    Raises
    ------
    ParameterMismatchError
        If the series have different alpha or beta parameters.

    Notes
    -----
    Converts to power basis, performs division, then converts back to Jacobi basis.

    Examples
    --------
    >>> a = jacobi_polynomial_p(torch.tensor([1.0, 2.0, 3.0]), alpha=0.0, beta=0.0)
    >>> b = jacobi_polynomial_p(torch.tensor([1.0, 1.0]), alpha=0.0, beta=0.0)
    >>> q, r = jacobi_polynomial_p_divmod(a, b)
    """
    # Check parameter compatibility
    if not torch.allclose(a.alpha, b.alpha) or not torch.allclose(
        a.beta, b.beta
    ):
        raise ParameterMismatchError(
            f"Cannot divide JacobiPolynomialP with alpha={a.alpha}, beta={a.beta} "
            f"by JacobiPolynomialP with alpha={b.alpha}, beta={b.beta}"
        )

    alpha = a.alpha
    beta = a.beta

    # Convert to power basis
    a_poly = jacobi_polynomial_p_to_polynomial(a)
    b_poly = jacobi_polynomial_p_to_polynomial(b)

    # Perform division in power basis
    q_poly, r_poly = polynomial_divmod(a_poly, b_poly)

    # Convert back to Jacobi basis
    q = polynomial_to_jacobi_polynomial_p(q_poly, alpha, beta)
    r = polynomial_to_jacobi_polynomial_p(r_poly, alpha, beta)

    return q, r

import warnings

import torch
from torch import Tensor

from ._jacobi_polynomial_p import JacobiPolynomialP
from ._jacobi_polynomial_p_antiderivative import (
    jacobi_polynomial_p_antiderivative,
)
from ._jacobi_polynomial_p_evaluate import jacobi_polynomial_p_evaluate


def jacobi_polynomial_p_integral(
    a: JacobiPolynomialP,
    lower: Tensor | float,
    upper: Tensor | float,
) -> Tensor:
    """Compute definite integral of Jacobi series.

    Computes integral_{lower}^{upper} a(x) dx.

    Parameters
    ----------
    a : JacobiPolynomialP
        Series to integrate.
    lower : Tensor
        Lower limit of integration.
    upper : Tensor
        Upper limit of integration.

    Returns
    -------
    Tensor
        Value of definite integral.

    Warnings
    --------
    UserWarning
        If integration bounds extend outside the natural domain [-1, 1].

    Notes
    -----
    Computed as F(upper) - F(lower) where F is the antiderivative.

    Examples
    --------
    >>> a = jacobi_polynomial_p(torch.tensor([1.0]), alpha=0.0, beta=0.0)  # constant 1
    >>> jacobi_polynomial_p_integral(a, torch.tensor(-1.0), torch.tensor(1.0))
    tensor(2.)
    """
    # Convert scalars to tensors
    if not isinstance(lower, Tensor):
        lower = torch.tensor(
            lower, dtype=a.coeffs.dtype, device=a.coeffs.device
        )
    if not isinstance(upper, Tensor):
        upper = torch.tensor(
            upper, dtype=a.coeffs.dtype, device=a.coeffs.device
        )

    domain = JacobiPolynomialP.DOMAIN

    if (lower < domain[0]).any() or (upper > domain[1]).any():
        warnings.warn(
            f"Integration bounds extend outside natural domain "
            f"[{domain[0]}, {domain[1]}] for JacobiPolynomialP. "
            f"Results may be numerically unstable.",
            stacklevel=2,
        )

    # Compute antiderivative with C=0
    antideriv = jacobi_polynomial_p_antiderivative(a, constant=0.0)

    # Evaluate at endpoints
    f_upper = jacobi_polynomial_p_evaluate(antideriv, upper)
    f_lower = jacobi_polynomial_p_evaluate(antideriv, lower)

    return f_upper - f_lower

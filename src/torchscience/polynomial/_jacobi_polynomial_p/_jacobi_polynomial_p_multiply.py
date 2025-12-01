import torch

from .._parameter_mismatch_error import ParameterMismatchError
from ._jacobi_polynomial_p import JacobiPolynomialP
from ._jacobi_polynomial_p_add import jacobi_polynomial_p_add
from ._jacobi_polynomial_p_mulx import jacobi_polynomial_p_mulx
from ._jacobi_polynomial_p_scale import jacobi_polynomial_p_scale


def jacobi_polynomial_p_multiply(
    a: JacobiPolynomialP,
    b: JacobiPolynomialP,
) -> JacobiPolynomialP:
    """Multiply two Jacobi series.

    Uses convolution in the Jacobi basis via repeated mulx operations.

    Parameters
    ----------
    a : JacobiPolynomialP
        First series with coefficients a_0, a_1, ..., a_m.
    b : JacobiPolynomialP
        Second series with coefficients b_0, b_1, ..., b_n.

    Returns
    -------
    JacobiPolynomialP
        Product series with degree at most m + n.

    Raises
    ------
    ParameterMismatchError
        If the series have different alpha or beta parameters.

    Notes
    -----
    The product of two Jacobi series of degrees m and n has degree m + n.
    The multiplication is performed by expanding one series in terms of
    the standard polynomial basis, multiplying by x repeatedly, and
    accumulating in Jacobi form.

    Examples
    --------
    >>> a = jacobi_polynomial_p(torch.tensor([1.0, 1.0]), alpha=0.0, beta=0.0)
    >>> b = jacobi_polynomial_p(torch.tensor([1.0, 1.0]), alpha=0.0, beta=0.0)
    >>> c = jacobi_polynomial_p_multiply(a, b)
    """
    # Check parameter compatibility
    if not torch.allclose(a.alpha, b.alpha) or not torch.allclose(
        a.beta, b.beta
    ):
        raise ParameterMismatchError(
            f"Cannot multiply JacobiPolynomialP with alpha={a.alpha}, beta={a.beta} "
            f"by JacobiPolynomialP with alpha={b.alpha}, beta={b.beta}"
        )

    a_coeffs = a.coeffs
    b_coeffs = b.coeffs
    alpha = a.alpha
    beta = a.beta

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    # Use the shorter polynomial for the outer loop
    if n_a > n_b:
        a_coeffs, b_coeffs = b_coeffs, a_coeffs
        n_a, n_b = n_b, n_a

    # Result: sum over k of a_k * (x^k in Jacobi form) * b
    # We build x^k * b iteratively using mulx

    # Start with x^0 * b = b
    x_power_b = JacobiPolynomialP(
        coeffs=b_coeffs.clone(), alpha=alpha.clone(), beta=beta.clone()
    )

    # Initialize result with a_0 * b
    result = jacobi_polynomial_p_scale(x_power_b, a_coeffs[..., 0:1])

    # Add contributions from higher powers
    for k in range(1, n_a):
        # Compute x * (x^{k-1} * b) = x^k * b
        x_power_b = jacobi_polynomial_p_mulx(x_power_b)

        # Add a_k * x^k * b to result
        contribution = jacobi_polynomial_p_scale(
            x_power_b, a_coeffs[..., k : k + 1]
        )
        result = jacobi_polynomial_p_add(result, contribution)

    return result

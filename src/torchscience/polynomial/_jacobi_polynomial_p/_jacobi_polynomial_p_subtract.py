import torch

from .._parameter_mismatch_error import ParameterMismatchError
from ._jacobi_polynomial_p import JacobiPolynomialP


def jacobi_polynomial_p_subtract(
    a: JacobiPolynomialP,
    b: JacobiPolynomialP,
) -> JacobiPolynomialP:
    """Subtract two Jacobi series.

    Parameters
    ----------
    a : JacobiPolynomialP
        First series.
    b : JacobiPolynomialP
        Second series.

    Returns
    -------
    JacobiPolynomialP
        Difference a - b.

    Raises
    ------
    ParameterMismatchError
        If the series have different alpha or beta parameters.

    Notes
    -----
    If the series have different degrees, the shorter one is zero-padded.

    Examples
    --------
    >>> a = jacobi_polynomial_p(torch.tensor([5.0, 7.0, 9.0]), alpha=0.5, beta=0.5)
    >>> b = jacobi_polynomial_p(torch.tensor([1.0, 2.0, 3.0]), alpha=0.5, beta=0.5)
    >>> c = jacobi_polynomial_p_subtract(a, b)
    >>> c.coeffs
    tensor([4., 5., 6.])
    """
    # Check parameter compatibility
    if not torch.allclose(a.alpha, b.alpha) or not torch.allclose(
        a.beta, b.beta
    ):
        raise ParameterMismatchError(
            f"Cannot subtract JacobiPolynomialP with alpha={a.alpha}, beta={a.beta} "
            f"from JacobiPolynomialP with alpha={b.alpha}, beta={b.beta}"
        )

    a_coeffs = a.coeffs
    b_coeffs = b.coeffs

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    if n_a == n_b:
        return JacobiPolynomialP(
            coeffs=a_coeffs - b_coeffs,
            alpha=a.alpha.clone(),
            beta=a.beta.clone(),
        )

    # Zero-pad the shorter series
    if n_a < n_b:
        pad_shape = list(a_coeffs.shape)
        pad_shape[-1] = n_b - n_a
        padding = torch.zeros(
            pad_shape, dtype=a_coeffs.dtype, device=a_coeffs.device
        )
        a_coeffs = torch.cat([a_coeffs, padding], dim=-1)
    else:
        pad_shape = list(b_coeffs.shape)
        pad_shape[-1] = n_a - n_b
        padding = torch.zeros(
            pad_shape, dtype=b_coeffs.dtype, device=b_coeffs.device
        )
        b_coeffs = torch.cat([b_coeffs, padding], dim=-1)

    return JacobiPolynomialP(
        coeffs=a_coeffs - b_coeffs, alpha=a.alpha.clone(), beta=a.beta.clone()
    )

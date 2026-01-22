import torch

from .._parameter_mismatch_error import ParameterMismatchError
from ._jacobi_polynomial_p import JacobiPolynomialP, jacobi_polynomial_p


def jacobi_polynomial_p_add(
    a: JacobiPolynomialP,
    b: JacobiPolynomialP,
) -> JacobiPolynomialP:
    """Add two Jacobi series.

    Parameters
    ----------
    a : JacobiPolynomialP
        First series.
    b : JacobiPolynomialP
        Second series.

    Returns
    -------
    JacobiPolynomialP
        Sum a + b.

    Raises
    ------
    ParameterMismatchError
        If the series have different alpha or beta parameters.

    Notes
    -----
    If the series have different degrees, the shorter one is zero-padded.

    Examples
    --------
    >>> a = jacobi_polynomial_p(torch.tensor([1.0, 2.0]), alpha=0.5, beta=0.5)
    >>> b = jacobi_polynomial_p(torch.tensor([3.0, 4.0, 5.0]), alpha=0.5, beta=0.5)
    >>> c = jacobi_polynomial_p_add(a, b)
    >>> c
    JacobiPolynomialP(tensor([4., 6., 5.]), alpha=tensor(0.5000), beta=tensor(0.5000))
    """
    # Check parameter compatibility
    if not torch.allclose(a.alpha, b.alpha) or not torch.allclose(
        a.beta, b.beta
    ):
        raise ParameterMismatchError(
            f"Cannot add JacobiPolynomialP with alpha={a.alpha}, beta={a.beta} "
            f"to JacobiPolynomialP with alpha={b.alpha}, beta={b.beta}"
        )

    # Get coefficients as plain tensors
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    if n_a == n_b:
        return jacobi_polynomial_p(
            a_coeffs + b_coeffs,
            a.alpha.clone(),
            a.beta.clone(),
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

    return jacobi_polynomial_p(
        a_coeffs + b_coeffs, a.alpha.clone(), a.beta.clone()
    )

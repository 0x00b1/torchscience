import torch
from torch import Tensor

from .._parameter_mismatch_error import ParameterMismatchError
from ._jacobi_polynomial_p import JacobiPolynomialP


def jacobi_polynomial_p_equal(
    a: JacobiPolynomialP,
    b: JacobiPolynomialP,
    tol: float = 1e-8,
) -> Tensor:
    """Check Jacobi series equality within tolerance.

    Parameters
    ----------
    a, b : JacobiPolynomialP
        Jacobi series to compare.
    tol : float
        Absolute tolerance for coefficient comparison.

    Returns
    -------
    Tensor
        Boolean tensor, shape matches broadcast of batch dims.

    Raises
    ------
    ParameterMismatchError
        If the series have different alpha or beta parameters.

    Notes
    -----
    Series of different lengths are compared by padding the shorter
    one with zeros.

    Examples
    --------
    >>> a = jacobi_polynomial_p(torch.tensor([1.0, 2.0, 3.0]), alpha=0.0, beta=0.0)
    >>> b = jacobi_polynomial_p(torch.tensor([1.0, 2.0, 3.0]), alpha=0.0, beta=0.0)
    >>> jacobi_polynomial_p_equal(a, b)
    tensor(True)
    """
    # Check parameter compatibility
    if not torch.allclose(a.alpha, b.alpha) or not torch.allclose(
        a.beta, b.beta
    ):
        raise ParameterMismatchError(
            f"Cannot compare JacobiPolynomialP with alpha={a.alpha}, beta={a.beta} "
            f"to JacobiPolynomialP with alpha={b.alpha}, beta={b.beta}"
        )

    # Get coefficients as plain tensors
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    # Pad to same length
    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    if n_a < n_b:
        padding = [0] * (2 * (a_coeffs.dim() - 1)) + [0, n_b - n_a]
        a_coeffs = torch.nn.functional.pad(a_coeffs, padding)
    elif n_b < n_a:
        padding = [0] * (2 * (b_coeffs.dim() - 1)) + [0, n_a - n_b]
        b_coeffs = torch.nn.functional.pad(b_coeffs, padding)

    # Check if all coefficients are within tolerance
    diff = (a_coeffs - b_coeffs).abs()

    # All coefficients must be within tolerance
    max_diff = diff.max(dim=-1)
    if isinstance(max_diff, tuple):
        max_diff = max_diff.values
    return max_diff <= tol

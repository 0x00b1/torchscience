import torch
from torch import Tensor

from ._laguerre_polynomial_l import LaguerrePolynomialL


def laguerre_polynomial_l_equal(
    a: LaguerrePolynomialL,
    b: LaguerrePolynomialL,
    tol: float = 1e-8,
) -> Tensor:
    """Check Laguerre series equality within tolerance.

    Parameters
    ----------
    a, b : LaguerrePolynomialL
        Laguerre series to compare.
    tol : float
        Absolute tolerance for coefficient comparison.

    Returns
    -------
    Tensor
        Boolean tensor, shape matches broadcast of batch dims.

    Notes
    -----
    Series of different lengths are compared by padding the shorter
    one with zeros.

    Examples
    --------
    >>> a = laguerre_polynomial_l(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = laguerre_polynomial_l(torch.tensor([1.0, 2.0, 3.0]))
    >>> laguerre_polynomial_l_equal(a, b)
    tensor(True)
    """
    a_coeffs = a.coeffs
    b_coeffs = b.coeffs

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

import torch
from torch import Tensor

from ._hermite_polynomial_he import HermitePolynomialHe


def hermite_polynomial_he_equal(
    a: HermitePolynomialHe,
    b: HermitePolynomialHe,
    tol: float = 1e-8,
) -> Tensor:
    """Check Probabilists' Hermite series equality within tolerance.

    Parameters
    ----------
    a, b : HermitePolynomialHe
        Hermite series to compare.
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
    >>> a = hermite_polynomial_he(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = hermite_polynomial_he(torch.tensor([1.0, 2.0, 3.0]))
    >>> hermite_polynomial_he_equal(a, b)
    tensor(True)
    """
    # Convert to plain tensors for operations
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

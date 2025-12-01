import torch
from torch import Tensor

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_equal(
    a: GegenbauerPolynomialC,
    b: GegenbauerPolynomialC,
    tol: float = 1e-8,
) -> Tensor:
    """Check Gegenbauer series equality within tolerance.

    Parameters
    ----------
    a, b : GegenbauerPolynomialC
        Gegenbauer series to compare.
    tol : float
        Absolute tolerance for coefficient and parameter comparison.

    Returns
    -------
    Tensor
        Boolean tensor, shape matches broadcast of batch dims.

    Notes
    -----
    Series of different lengths are compared by padding the shorter
    one with zeros. Both coefficients and lambda parameters must match.

    Examples
    --------
    >>> a = gegenbauer_polynomial_c(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(1.0))
    >>> b = gegenbauer_polynomial_c(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(1.0))
    >>> gegenbauer_polynomial_c_equal(a, b)
    tensor(True)
    """
    # Check parameter equality
    if not torch.allclose(a.lambda_, b.lambda_, atol=tol):
        return torch.tensor(False, device=a.coeffs.device)

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

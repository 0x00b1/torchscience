import torch
from torch import Tensor

from ._polynomial import Polynomial


def polynomial_equal(
    p: Polynomial,
    q: Polynomial,
    tol: float = 1e-8,
) -> Tensor:
    """Check polynomial equality within tolerance.

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to compare.
    tol : float
        Absolute tolerance for coefficient comparison.

    Returns
    -------
    Tensor
        Boolean tensor, shape matches broadcast of batch dims.
    """
    # p and q ARE the coefficient tensors now
    # Work with plain tensor views
    p_coeffs = p.as_subclass(Tensor)
    q_coeffs = q.as_subclass(Tensor)

    # Pad to same length
    n_p = p_coeffs.shape[-1]
    n_q = q_coeffs.shape[-1]

    if n_p < n_q:
        padding = [0] * (2 * (p_coeffs.dim() - 1)) + [0, n_q - n_p]
        p_coeffs = torch.nn.functional.pad(p_coeffs, padding)
    elif n_q < n_p:
        padding = [0] * (2 * (q_coeffs.dim() - 1)) + [0, n_p - n_q]
        q_coeffs = torch.nn.functional.pad(q_coeffs, padding)

    # Check if all coefficients are within tolerance
    diff = (p_coeffs - q_coeffs).abs()

    # All coefficients must be within tolerance
    max_diff = diff.max(dim=-1)
    if isinstance(max_diff, tuple):
        max_diff = max_diff.values
    return max_diff <= tol

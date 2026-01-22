import torch

from ._polynomial import Polynomial, polynomial


def polynomial_trim(p: Polynomial, tol: float = 0.0) -> Polynomial:
    """Remove trailing near-zero coefficients.

    Parameters
    ----------
    p : Polynomial
        Input polynomial.
    tol : float
        Tolerance for considering coefficient as zero.

    Returns
    -------
    Polynomial
        Trimmed polynomial with at least one coefficient.

    Notes
    -----
    For batched polynomials, this trims based on the maximum absolute
    value across the batch for each coefficient position.
    """
    # p IS the coefficient tensor now
    n = p.shape[-1]

    if n <= 1:
        return p

    # Find the last non-zero coefficient
    # For batched case, a coefficient position is non-zero if any batch element is non-zero
    if p.dim() > 1:
        abs_coeffs = p.abs()
        # Max over batch dimensions
        max_abs = abs_coeffs
        for _ in range(p.dim() - 1):
            max_abs = max_abs.max(dim=0).values
    else:
        max_abs = p.abs()

    # Find last position > tol
    mask = max_abs > tol
    if not mask.any():
        # All zeros, return single zero coefficient
        return polynomial(
            torch.zeros(*p.shape[:-1], 1, dtype=p.dtype, device=p.device)
        )

    # Find last True position
    indices = torch.arange(n, device=p.device)
    last_nonzero = indices[mask].max().item()

    # Keep coefficients up to and including last_nonzero
    return polynomial(p[..., : last_nonzero + 1])

import torch

from ._laguerre_polynomial_l import (
    LaguerrePolynomialL,
    laguerre_polynomial_l,
)


def laguerre_polynomial_l_trim(
    p: LaguerrePolynomialL,
    tol: float = 0.0,
) -> LaguerrePolynomialL:
    """Remove trailing near-zero coefficients.

    Parameters
    ----------
    p : LaguerrePolynomialL
        Input Laguerre series.
    tol : float
        Tolerance for considering coefficient as zero.

    Returns
    -------
    LaguerrePolynomialL
        Trimmed series with at least one coefficient.

    Notes
    -----
    For batched series, this trims based on the maximum absolute
    value across the batch for each coefficient position.

    Examples
    --------
    >>> c = laguerre_polynomial_l(torch.tensor([1.0, 2.0, 0.0, 0.0]))
    >>> t = laguerre_polynomial_l_trim(c)
    >>> t
    LaguerrePolynomialL(tensor([1., 2.]))
    """
    coeffs = p.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    if n <= 1:
        return p

    # Find the last non-zero coefficient
    # For batched case, a coefficient position is non-zero if any batch element is non-zero
    if coeffs.dim() > 1:
        abs_coeffs = coeffs.abs()
        # Max over batch dimensions
        max_abs = abs_coeffs
        for _ in range(coeffs.dim() - 1):
            max_abs = max_abs.max(dim=0).values
    else:
        max_abs = coeffs.abs()

    # Find last position > tol
    mask = max_abs > tol
    if not mask.any():
        # All zeros, return single zero coefficient
        return laguerre_polynomial_l(
            torch.zeros(
                *coeffs.shape[:-1], 1, dtype=coeffs.dtype, device=coeffs.device
            )
        )

    # Find last True position
    indices = torch.arange(n, device=coeffs.device)
    last_nonzero = indices[mask].max().item()

    # Keep coefficients up to and including last_nonzero
    return laguerre_polynomial_l(coeffs[..., : last_nonzero + 1])

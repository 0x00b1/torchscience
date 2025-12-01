import torch

from ._legendre_polynomial_p import LegendrePolynomialP


def legendre_polynomial_p_trim(
    p: LegendrePolynomialP,
    tol: float = 0.0,
) -> LegendrePolynomialP:
    """Remove trailing near-zero coefficients.

    Parameters
    ----------
    p : LegendrePolynomialP
        Input Legendre series.
    tol : float
        Tolerance for considering coefficient as zero.

    Returns
    -------
    LegendrePolynomialP
        Trimmed series with at least one coefficient.

    Notes
    -----
    For batched series, this trims based on the maximum absolute
    value across the batch for each coefficient position.

    Examples
    --------
    >>> c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 0.0, 0.0]))
    >>> t = legendre_polynomial_p_trim(c)
    >>> t.coeffs
    tensor([1., 2.])
    """
    coeffs = p.coeffs
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
        return LegendrePolynomialP(
            coeffs=torch.zeros(
                *coeffs.shape[:-1], 1, dtype=coeffs.dtype, device=coeffs.device
            )
        )

    # Find last True position
    indices = torch.arange(n, device=coeffs.device)
    last_nonzero = indices[mask].max().item()

    # Keep coefficients up to and including last_nonzero
    return LegendrePolynomialP(coeffs=coeffs[..., : last_nonzero + 1])

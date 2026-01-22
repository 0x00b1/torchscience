import torch

from ._hermite_polynomial_he import (
    HermitePolynomialHe,
    hermite_polynomial_he,
)


def hermite_polynomial_he_trim(
    p: HermitePolynomialHe,
    tol: float = 0.0,
) -> HermitePolynomialHe:
    """Remove trailing near-zero coefficients.

    Parameters
    ----------
    p : HermitePolynomialHe
        Input Hermite series.
    tol : float
        Tolerance for considering coefficient as zero.

    Returns
    -------
    HermitePolynomialHe
        Trimmed series with at least one coefficient.

    Notes
    -----
    For batched series, this trims based on the maximum absolute
    value across the batch for each coefficient position.

    Examples
    --------
    >>> c = hermite_polynomial_he(torch.tensor([1.0, 2.0, 0.0, 0.0]))
    >>> t = hermite_polynomial_he_trim(c)
    >>> t
    HermitePolynomialHe(tensor([1., 2.]))
    """
    # Convert to plain tensor for operations
    coeffs = p.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    if n <= 1:
        return hermite_polynomial_he(coeffs.clone())

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
        return hermite_polynomial_he(
            torch.zeros(
                *coeffs.shape[:-1], 1, dtype=coeffs.dtype, device=coeffs.device
            )
        )

    # Find last True position
    indices = torch.arange(n, device=coeffs.device)
    last_nonzero = indices[mask].max().item()

    # Keep coefficients up to and including last_nonzero
    return hermite_polynomial_he(coeffs[..., : last_nonzero + 1].clone())

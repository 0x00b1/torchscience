import torch

from ._jacobi_polynomial_p import JacobiPolynomialP, jacobi_polynomial_p


def jacobi_polynomial_p_trim(
    p: JacobiPolynomialP,
    tol: float = 0.0,
) -> JacobiPolynomialP:
    """Remove trailing near-zero coefficients.

    Parameters
    ----------
    p : JacobiPolynomialP
        Input Jacobi series.
    tol : float
        Tolerance for considering coefficient as zero.

    Returns
    -------
    JacobiPolynomialP
        Trimmed series with at least one coefficient.

    Notes
    -----
    For batched series, this trims based on the maximum absolute
    value across the batch for each coefficient position.

    Examples
    --------
    >>> c = jacobi_polynomial_p(torch.tensor([1.0, 2.0, 0.0, 0.0]), alpha=0.0, beta=0.0)
    >>> t = jacobi_polynomial_p_trim(c)
    >>> t
    JacobiPolynomialP(tensor([1., 2.]), alpha=tensor(0.), beta=tensor(0.))
    """
    # Get coefficients as plain tensor
    coeffs = p.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    if n <= 1:
        return jacobi_polynomial_p(
            coeffs.clone(), p.alpha.clone(), p.beta.clone()
        )

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
        return jacobi_polynomial_p(
            torch.zeros(
                *coeffs.shape[:-1], 1, dtype=coeffs.dtype, device=coeffs.device
            ),
            p.alpha.clone(),
            p.beta.clone(),
        )

    # Find last True position
    indices = torch.arange(n, device=coeffs.device)
    last_nonzero = indices[mask].max().item()

    # Keep coefficients up to and including last_nonzero
    return jacobi_polynomial_p(
        coeffs[..., : last_nonzero + 1],
        p.alpha.clone(),
        p.beta.clone(),
    )

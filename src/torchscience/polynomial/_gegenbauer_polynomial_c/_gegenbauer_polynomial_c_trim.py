import torch
from torch import Tensor

from ._gegenbauer_polynomial_c import (
    GegenbauerPolynomialC,
    gegenbauer_polynomial_c,
)


def gegenbauer_polynomial_c_trim(
    p: GegenbauerPolynomialC,
    tol: float = 0.0,
) -> GegenbauerPolynomialC:
    """Remove trailing near-zero coefficients.

    Parameters
    ----------
    p : GegenbauerPolynomialC
        Input Gegenbauer series.
    tol : float
        Tolerance for considering coefficient as zero.

    Returns
    -------
    GegenbauerPolynomialC
        Trimmed series with at least one coefficient.

    Notes
    -----
    For batched series, this trims based on the maximum absolute
    value across the batch for each coefficient position.

    Examples
    --------
    >>> c = gegenbauer_polynomial_c(
    ...     torch.tensor([1.0, 2.0, 0.0, 0.0]), torch.tensor(1.0)
    ... )
    >>> t = gegenbauer_polynomial_c_trim(c)
    >>> t
    GegenbauerPolynomialC(tensor([1., 2.]), lambda_=tensor(1.))
    """
    # Get coefficients as plain tensor
    coeffs = p.as_subclass(Tensor)
    n = coeffs.shape[-1]

    if n <= 1:
        return gegenbauer_polynomial_c(coeffs.clone(), p.lambda_)

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
        return gegenbauer_polynomial_c(
            torch.zeros(
                *coeffs.shape[:-1], 1, dtype=coeffs.dtype, device=coeffs.device
            ),
            p.lambda_,
        )

    # Find last True position
    indices = torch.arange(n, device=coeffs.device)
    last_nonzero = indices[mask].max().item()

    # Keep coefficients up to and including last_nonzero
    return gegenbauer_polynomial_c(
        coeffs[..., : last_nonzero + 1].clone(),
        p.lambda_,
    )

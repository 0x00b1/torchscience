import torch

from ._chebyshev_polynomial_w import ChebyshevPolynomialW


def chebyshev_polynomial_w_trim(
    a: ChebyshevPolynomialW,
    tol: float = 0.0,
) -> ChebyshevPolynomialW:
    """Remove trailing coefficients smaller than tol.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Input series.
    tol : float, optional
        Tolerance for trimming. Coefficients with absolute value
        less than or equal to tol are removed. Default is 0.0.

    Returns
    -------
    ChebyshevPolynomialW
        Trimmed series with at least one coefficient.

    Notes
    -----
    The result always has at least one coefficient (the constant term),
    even if it is zero.

    Examples
    --------
    >>> a = ChebyshevPolynomialW(coeffs=torch.tensor([1.0, 2.0, 0.0, 0.0]))
    >>> chebyshev_polynomial_w_trim(a)  # Removes trailing zeros
    """
    coeffs = a.coeffs

    # Find last significant coefficient
    # Work from the end, find first coefficient with |c| > tol
    n = coeffs.shape[-1]

    # Find index of last significant coefficient
    last_idx = 0
    for i in range(n - 1, -1, -1):
        if torch.abs(coeffs[..., i]).max() > tol:
            last_idx = i
            break

    # Always keep at least one coefficient
    new_coeffs = coeffs[..., : last_idx + 1]

    return ChebyshevPolynomialW(coeffs=new_coeffs)

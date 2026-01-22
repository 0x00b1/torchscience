import torch

from ._chebyshev_polynomial_v import ChebyshevPolynomialV


def chebyshev_polynomial_v_equal(
    a: ChebyshevPolynomialV,
    b: ChebyshevPolynomialV,
    tol: float = 0.0,
) -> bool:
    """Check if two Chebyshev V series are equal.

    Parameters
    ----------
    a : ChebyshevPolynomialV
        First series.
    b : ChebyshevPolynomialV
        Second series.
    tol : float, optional
        Tolerance for comparison. Default is 0.0 (exact).

    Returns
    -------
    bool
        True if series are equal (within tolerance).

    Notes
    -----
    Series of different lengths are compared by zero-padding the shorter one.

    Examples
    --------
    >>> a = chebyshev_polynomial_v(torch.tensor([1.0, 2.0]))
    >>> b = chebyshev_polynomial_v(torch.tensor([1.0, 2.0, 0.0]))
    >>> chebyshev_polynomial_v_equal(a, b)
    True
    """
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    # Pad to same length
    if n_a < n_b:
        padding = torch.zeros(
            n_b - n_a, dtype=a_coeffs.dtype, device=a_coeffs.device
        )
        a_coeffs = torch.cat([a_coeffs, padding], dim=-1)
    elif n_b < n_a:
        padding = torch.zeros(
            n_a - n_b, dtype=b_coeffs.dtype, device=b_coeffs.device
        )
        b_coeffs = torch.cat([b_coeffs, padding], dim=-1)

    diff = torch.abs(a_coeffs - b_coeffs)
    return bool(diff.max() <= tol)

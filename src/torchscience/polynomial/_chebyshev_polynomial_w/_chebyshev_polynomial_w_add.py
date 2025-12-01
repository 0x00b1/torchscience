import torch

from ._chebyshev_polynomial_w import ChebyshevPolynomialW


def chebyshev_polynomial_w_add(
    a: ChebyshevPolynomialW,
    b: ChebyshevPolynomialW,
) -> ChebyshevPolynomialW:
    """Add two Chebyshev W series.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        First series.
    b : ChebyshevPolynomialW
        Second series.

    Returns
    -------
    ChebyshevPolynomialW
        Sum a + b.

    Notes
    -----
    If the series have different degrees, the shorter one is zero-padded.

    Examples
    --------
    >>> a = chebyshev_polynomial_w(torch.tensor([1.0, 2.0]))
    >>> b = chebyshev_polynomial_w(torch.tensor([3.0, 4.0, 5.0]))
    >>> c = chebyshev_polynomial_w_add(a, b)
    >>> c.coeffs
    tensor([4., 6., 5.])
    """
    a_coeffs = a.coeffs
    b_coeffs = b.coeffs

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    if n_a == n_b:
        return ChebyshevPolynomialW(coeffs=a_coeffs + b_coeffs)

    # Zero-pad the shorter series
    if n_a < n_b:
        pad_shape = list(a_coeffs.shape)
        pad_shape[-1] = n_b - n_a
        padding = torch.zeros(
            pad_shape, dtype=a_coeffs.dtype, device=a_coeffs.device
        )
        a_coeffs = torch.cat([a_coeffs, padding], dim=-1)
    else:
        pad_shape = list(b_coeffs.shape)
        pad_shape[-1] = n_a - n_b
        padding = torch.zeros(
            pad_shape, dtype=b_coeffs.dtype, device=b_coeffs.device
        )
        b_coeffs = torch.cat([b_coeffs, padding], dim=-1)

    return ChebyshevPolynomialW(coeffs=a_coeffs + b_coeffs)

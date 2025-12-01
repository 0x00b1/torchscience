import torch
from torch import Tensor

from ._chebyshev_polynomial_w import ChebyshevPolynomialW


def chebyshev_polynomial_w_equal(
    a: ChebyshevPolynomialW,
    b: ChebyshevPolynomialW,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tensor:
    """Check if two Chebyshev W series are equal within tolerance.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        First series.
    b : ChebyshevPolynomialW
        Second series.
    rtol : float, optional
        Relative tolerance. Default is 1e-5.
    atol : float, optional
        Absolute tolerance. Default is 1e-8.

    Returns
    -------
    Tensor
        Boolean tensor, True if series are approximately equal.

    Notes
    -----
    Series of different lengths are compared by zero-padding
    the shorter one.

    Examples
    --------
    >>> a = ChebyshevPolynomialW(coeffs=torch.tensor([1.0, 2.0]))
    >>> b = ChebyshevPolynomialW(coeffs=torch.tensor([1.0, 2.0, 0.0]))
    >>> chebyshev_polynomial_w_equal(a, b)
    tensor(True)
    """
    c1 = a.coeffs
    c2 = b.coeffs

    n1 = c1.shape[-1]
    n2 = c2.shape[-1]

    # Pad shorter to match longer
    if n1 < n2:
        pad = torch.zeros(
            *c1.shape[:-1], n2 - n1, dtype=c1.dtype, device=c1.device
        )
        c1 = torch.cat([c1, pad], dim=-1)
    elif n2 < n1:
        pad = torch.zeros(
            *c2.shape[:-1], n1 - n2, dtype=c2.dtype, device=c2.device
        )
        c2 = torch.cat([c2, pad], dim=-1)

    return torch.tensor(torch.allclose(c1, c2, rtol=rtol, atol=atol))

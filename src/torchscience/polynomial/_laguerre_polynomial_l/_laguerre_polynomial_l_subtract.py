import torch

from ._laguerre_polynomial_l import (
    LaguerrePolynomialL,
    laguerre_polynomial_l,
)


def laguerre_polynomial_l_subtract(
    a: LaguerrePolynomialL,
    b: LaguerrePolynomialL,
) -> LaguerrePolynomialL:
    """Subtract two Laguerre series.

    Parameters
    ----------
    a : LaguerrePolynomialL
        First series.
    b : LaguerrePolynomialL
        Second series.

    Returns
    -------
    LaguerrePolynomialL
        Difference a - b.

    Notes
    -----
    If the series have different degrees, the shorter one is zero-padded.

    Examples
    --------
    >>> a = laguerre_polynomial_l(torch.tensor([5.0, 7.0, 9.0]))
    >>> b = laguerre_polynomial_l(torch.tensor([1.0, 2.0, 3.0]))
    >>> c = laguerre_polynomial_l_subtract(a, b)
    >>> c
    LaguerrePolynomialL(tensor([4., 5., 6.]))
    """
    n_a = a.shape[-1]
    n_b = b.shape[-1]

    if n_a == n_b:
        result = torch.Tensor.sub(a, b)
        return laguerre_polynomial_l(result)

    # Zero-pad the shorter series
    if n_a < n_b:
        pad_shape = list(a.shape)
        pad_shape[-1] = n_b - n_a
        padding = torch.zeros(pad_shape, dtype=a.dtype, device=a.device)
        a_padded = torch.cat([a.as_subclass(torch.Tensor), padding], dim=-1)
        result = a_padded - b.as_subclass(torch.Tensor)
    else:
        pad_shape = list(b.shape)
        pad_shape[-1] = n_a - n_b
        padding = torch.zeros(pad_shape, dtype=b.dtype, device=b.device)
        b_padded = torch.cat([b.as_subclass(torch.Tensor), padding], dim=-1)
        result = a.as_subclass(torch.Tensor) - b_padded

    return laguerre_polynomial_l(result)

import torch

from ._chebyshev_polynomial_t import (
    ChebyshevPolynomialT,
    chebyshev_polynomial_t,
)


def chebyshev_polynomial_t_subtract(
    a: ChebyshevPolynomialT,
    b: ChebyshevPolynomialT,
) -> ChebyshevPolynomialT:
    """Subtract two Chebyshev series.

    Parameters
    ----------
    a : ChebyshevPolynomialT
        First series.
    b : ChebyshevPolynomialT
        Second series.

    Returns
    -------
    ChebyshevPolynomialT
        Difference a - b.

    Notes
    -----
    If the series have different degrees, the shorter one is zero-padded.

    Examples
    --------
    >>> a = chebyshev_polynomial_t(torch.tensor([5.0, 7.0, 9.0]))
    >>> b = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
    >>> c = chebyshev_polynomial_t_subtract(a, b)
    >>> c
    ChebyshevPolynomialT(tensor([4., 5., 6.]))
    """
    n_a = a.shape[-1]
    n_b = b.shape[-1]

    if n_a == n_b:
        result = torch.Tensor.sub(a, b)
        return chebyshev_polynomial_t(result)

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

    return chebyshev_polynomial_t(result)

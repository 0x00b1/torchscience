import torch

from ._chebyshev_polynomial_w import (
    ChebyshevPolynomialW,
    chebyshev_polynomial_w,
)


def chebyshev_polynomial_w_subtract(
    a: ChebyshevPolynomialW,
    b: ChebyshevPolynomialW,
) -> ChebyshevPolynomialW:
    """Subtract two Chebyshev W series.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        First series.
    b : ChebyshevPolynomialW
        Second series.

    Returns
    -------
    ChebyshevPolynomialW
        Difference a - b.

    Notes
    -----
    If the series have different degrees, the shorter one is zero-padded.

    Examples
    --------
    >>> a = chebyshev_polynomial_w(torch.tensor([5.0, 7.0, 9.0]))
    >>> b = chebyshev_polynomial_w(torch.tensor([1.0, 2.0, 3.0]))
    >>> c = chebyshev_polynomial_w_subtract(a, b)
    >>> c
    ChebyshevPolynomialW(tensor([4., 5., 6.]))
    """
    n_a = a.shape[-1]
    n_b = b.shape[-1]

    if n_a == n_b:
        # Use Tensor subtraction to get raw tensor, then wrap
        result = torch.Tensor.sub(a, b)
        return chebyshev_polynomial_w(result)

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

    return chebyshev_polynomial_w(result)

import torch

from ._chebyshev_polynomial_u import (
    ChebyshevPolynomialU,
    chebyshev_polynomial_u,
)


def chebyshev_polynomial_u_add(
    a: ChebyshevPolynomialU,
    b: ChebyshevPolynomialU,
) -> ChebyshevPolynomialU:
    """Add two Chebyshev U series.

    Parameters
    ----------
    a : ChebyshevPolynomialU
        First series.
    b : ChebyshevPolynomialU
        Second series.

    Returns
    -------
    ChebyshevPolynomialU
        Sum a + b.

    Notes
    -----
    If the series have different degrees, the shorter one is zero-padded.

    Examples
    --------
    >>> a = chebyshev_polynomial_u(torch.tensor([1.0, 2.0]))
    >>> b = chebyshev_polynomial_u(torch.tensor([3.0, 4.0, 5.0]))
    >>> c = chebyshev_polynomial_u_add(a, b)
    >>> c
    ChebyshevPolynomialU(tensor([4., 6., 5.]))
    """
    n_a = a.shape[-1]
    n_b = b.shape[-1]

    if n_a == n_b:
        # Use Tensor addition to get raw tensor, then wrap
        result = torch.Tensor.add(a, b)
        return chebyshev_polynomial_u(result)

    # Zero-pad the shorter series
    if n_a < n_b:
        pad_shape = list(a.shape)
        pad_shape[-1] = n_b - n_a
        padding = torch.zeros(pad_shape, dtype=a.dtype, device=a.device)
        a_padded = torch.cat([a.as_subclass(torch.Tensor), padding], dim=-1)
        result = a_padded + b.as_subclass(torch.Tensor)
    else:
        pad_shape = list(b.shape)
        pad_shape[-1] = n_a - n_b
        padding = torch.zeros(pad_shape, dtype=b.dtype, device=b.device)
        b_padded = torch.cat([b.as_subclass(torch.Tensor), padding], dim=-1)
        result = a.as_subclass(torch.Tensor) + b_padded

    return chebyshev_polynomial_u(result)

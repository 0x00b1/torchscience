import torch

from ._hermite_polynomial_h import (
    HermitePolynomialH,
    hermite_polynomial_h,
)


def hermite_polynomial_h_add(
    a: HermitePolynomialH,
    b: HermitePolynomialH,
) -> HermitePolynomialH:
    """Add two Physicists' Hermite series.

    Parameters
    ----------
    a : HermitePolynomialH
        First series.
    b : HermitePolynomialH
        Second series.

    Returns
    -------
    HermitePolynomialH
        Sum a + b.

    Notes
    -----
    If the series have different degrees, the shorter one is zero-padded.

    Examples
    --------
    >>> a = hermite_polynomial_h(torch.tensor([1.0, 2.0]))
    >>> b = hermite_polynomial_h(torch.tensor([3.0, 4.0, 5.0]))
    >>> c = hermite_polynomial_h_add(a, b)
    >>> c
    HermitePolynomialH(tensor([4., 6., 5.]))
    """
    n_a = a.shape[-1]
    n_b = b.shape[-1]

    if n_a == n_b:
        # Use Tensor addition to get raw tensor, then wrap
        result = torch.Tensor.add(a, b)
        return hermite_polynomial_h(result)

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

    return hermite_polynomial_h(result)

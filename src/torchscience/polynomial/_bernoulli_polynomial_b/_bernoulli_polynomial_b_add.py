"""Add two Bernoulli polynomial series."""

import torch

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
    bernoulli_polynomial_b,
)


def bernoulli_polynomial_b_add(
    a: BernoulliPolynomialB,
    b: BernoulliPolynomialB,
) -> BernoulliPolynomialB:
    """Add two Bernoulli polynomial series.

    Parameters
    ----------
    a : BernoulliPolynomialB
        First series.
    b : BernoulliPolynomialB
        Second series.

    Returns
    -------
    BernoulliPolynomialB
        Sum a + b.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
    >>> b = bernoulli_polynomial_b(torch.tensor([3.0, 4.0]))
    >>> bernoulli_polynomial_b_add(a, b)
    BernoulliPolynomialB(tensor([4., 6.]))
    """
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]
    n_max = max(n_a, n_b)

    # Pad shorter series with zeros
    if n_a < n_max:
        pad_shape = list(a_coeffs.shape)
        pad_shape[-1] = n_max - n_a
        a_coeffs = torch.cat(
            [
                a_coeffs,
                torch.zeros(
                    pad_shape, dtype=a_coeffs.dtype, device=a_coeffs.device
                ),
            ],
            dim=-1,
        )

    if n_b < n_max:
        pad_shape = list(b_coeffs.shape)
        pad_shape[-1] = n_max - n_b
        b_coeffs = torch.cat(
            [
                b_coeffs,
                torch.zeros(
                    pad_shape, dtype=b_coeffs.dtype, device=b_coeffs.device
                ),
            ],
            dim=-1,
        )

    return bernoulli_polynomial_b(a_coeffs + b_coeffs)

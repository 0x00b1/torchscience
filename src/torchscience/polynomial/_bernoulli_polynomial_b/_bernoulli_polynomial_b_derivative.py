"""Derivative of Bernoulli polynomial series."""

import torch

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
    bernoulli_polynomial_b,
)


def bernoulli_polynomial_b_derivative(
    a: BernoulliPolynomialB,
    order: int = 1,
) -> BernoulliPolynomialB:
    r"""Compute derivative of Bernoulli polynomial series.

    Uses the fundamental property of Bernoulli polynomials:

    .. math::

        B'_n(x) = n \cdot B_{n-1}(x)

    For f(x) = sum_{k=0}^{n} c[k] * B_k(x), the derivative is:

    .. math::

        f'(x) = \sum_{k=1}^{n} c[k] \cdot k \cdot B_{k-1}(x)
              = \sum_{j=0}^{n-1} c[j+1] \cdot (j+1) \cdot B_j(x)

    Parameters
    ----------
    a : BernoulliPolynomialB
        Series to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    BernoulliPolynomialB
        Derivative series.

    Raises
    ------
    ValueError
        If order is negative.

    Notes
    -----
    The degree decreases by 1 for each differentiation.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([0.0, 0.0, 1.0]))  # B_2(x)
    >>> da = bernoulli_polynomial_b_derivative(a)
    >>> # B'_2(x) = 2 * B_1(x), so coeffs should be [0.0, 2.0]
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    coeffs = a.as_subclass(torch.Tensor)

    if order == 0:
        return bernoulli_polynomial_b(coeffs.clone())

    # Apply derivative 'order' times
    for _ in range(order):
        n = coeffs.shape[-1]

        if n <= 1:
            # Derivative of constant is zero
            result_shape = list(coeffs.shape)
            result_shape[-1] = 1
            coeffs = torch.zeros(
                result_shape, dtype=coeffs.dtype, device=coeffs.device
            )
            continue

        # For f(x) = sum c[k] * B_k(x)
        # f'(x) = sum c[k] * k * B_{k-1}(x)
        #       = sum_{j=0}^{n-2} c[j+1] * (j+1) * B_j(x)
        new_n = n - 1
        result_shape = list(coeffs.shape)
        result_shape[-1] = new_n
        new_coeffs = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # k goes from 1 to n-1, j = k-1 goes from 0 to n-2
        for j in range(new_n):
            k = j + 1
            new_coeffs[..., j] = coeffs[..., k] * k

        coeffs = new_coeffs

    return bernoulli_polynomial_b(coeffs)

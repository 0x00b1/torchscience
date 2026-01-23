"""Antiderivative of Bernoulli polynomial series."""

import torch

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
    bernoulli_polynomial_b,
)


def bernoulli_polynomial_b_antiderivative(
    a: BernoulliPolynomialB,
    order: int = 1,
    constant: float = 0.0,
) -> BernoulliPolynomialB:
    r"""Compute antiderivative of Bernoulli polynomial series.

    Uses the inverse of the derivative property:

    .. math::

        \int B_n(x) dx = \frac{B_{n+1}(x)}{n+1} + C

    For f(x) = sum_{k=0}^{n} c[k] * B_k(x), the antiderivative is:

    .. math::

        \int f(x) dx = C \cdot B_0(x) + \sum_{k=0}^{n} \frac{c[k]}{k+1} B_{k+1}(x)

    Parameters
    ----------
    a : BernoulliPolynomialB
        Series to integrate.
    order : int, optional
        Order of antiderivative. Default is 1.
    constant : float, optional
        Integration constant. Default is 0.0.

    Returns
    -------
    BernoulliPolynomialB
        Antiderivative series.

    Raises
    ------
    ValueError
        If order is negative.

    Notes
    -----
    The degree increases by 1 for each integration.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([0.0, 2.0]))  # 2*B_1(x)
    >>> A = bernoulli_polynomial_b_antiderivative(a)
    >>> # integral of 2*B_1(x) = 2 * B_2(x)/2 + C = B_2(x) + C
    >>> # coeffs should be [C, 0.0, 1.0]
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    coeffs = a.as_subclass(torch.Tensor)

    if order == 0:
        return bernoulli_polynomial_b(coeffs.clone())

    # Apply antiderivative 'order' times
    for _ in range(order):
        n = coeffs.shape[-1]

        # New size is n + 1
        new_n = n + 1
        result_shape = list(coeffs.shape)
        result_shape[-1] = new_n
        new_coeffs = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # Integration constant goes to B_0 coefficient
        new_coeffs[..., 0] = constant

        # For each c[k] * B_k(x), antiderivative is c[k]/(k+1) * B_{k+1}(x)
        for k in range(n):
            new_coeffs[..., k + 1] = coeffs[..., k] / (k + 1)

        coeffs = new_coeffs
        constant = 0.0  # Only first integration gets the constant

    return bernoulli_polynomial_b(coeffs)

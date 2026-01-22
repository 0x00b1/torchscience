import torch

from ._legendre_polynomial_p import (
    LegendrePolynomialP,
    legendre_polynomial_p,
)


def legendre_polynomial_p_derivative(
    a: LegendrePolynomialP,
    order: int = 1,
) -> LegendrePolynomialP:
    """Compute derivative of Legendre series.

    Uses the recurrence relation:
        P'_n = (2n-1)*P_{n-1} + P'_{n-2}

    which leads to the algorithm (matching numpy's legder):
        der[j-1] = (2j-1)*c[j], c[j-2] += c[j]  for j = n, n-1, ..., 3
        der[1] = 3*c[2]
        der[0] = c[1]

    Parameters
    ----------
    a : LegendrePolynomialP
        Series to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    LegendrePolynomialP
        Derivative series.

    Notes
    -----
    The degree decreases by 1 for each differentiation.

    Examples
    --------
    >>> a = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
    >>> da = legendre_polynomial_p_derivative(a)
    >>> da  # d/dx P_1 = 1 = P_0
    LegendrePolynomialP(tensor([1.]))
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    coeffs = a.as_subclass(torch.Tensor)

    if order == 0:
        return legendre_polynomial_p(coeffs.clone())

    coeffs = coeffs.clone()  # Will be modified during iteration
    n = coeffs.shape[-1]

    # Apply derivative 'order' times
    for _ in range(order):
        if n <= 1:
            # Derivative of constant is zero
            result_shape = list(coeffs.shape)
            result_shape[-1] = 1
            coeffs = torch.zeros(
                result_shape, dtype=coeffs.dtype, device=coeffs.device
            )
            n = 1
            continue

        # Result has n-1 coefficients (degree decreases by 1)
        new_n = n - 1
        result_shape = list(coeffs.shape)
        result_shape[-1] = new_n
        der = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # Algorithm matching numpy's legder:
        # For j from n-1 (degree) down to 2:
        #   der[j-1] = (2j-1) * coeffs[j]
        #   coeffs[j-2] += coeffs[j]  (accumulate for lower terms)
        # Then:
        #   der[1] = 3 * coeffs[2]  (if new_n > 1)
        #   der[0] = coeffs[1]

        # Note: j is the degree index, so j goes from n-1 down to 2
        for j in range(n - 1, 2, -1):
            der[..., j - 1] = (2.0 * j - 1.0) * coeffs[..., j]
            # Accumulate: coeffs[j-2] += coeffs[j]
            coeffs[..., j - 2] = coeffs[..., j - 2] + coeffs[..., j]

        if new_n > 1:
            der[..., 1] = 3.0 * coeffs[..., 2]

        der[..., 0] = coeffs[..., 1]

        coeffs = der
        n = new_n

    return legendre_polynomial_p(coeffs)

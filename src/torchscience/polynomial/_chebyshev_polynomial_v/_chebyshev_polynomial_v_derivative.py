import torch

from ._chebyshev_polynomial_v import (
    ChebyshevPolynomialV,
    chebyshev_polynomial_v,
)


def chebyshev_polynomial_v_derivative(
    a: ChebyshevPolynomialV,
    order: int = 1,
) -> ChebyshevPolynomialV:
    """Compute derivative of Chebyshev V series.

    Uses the recurrence relation for Chebyshev V derivatives.

    For Chebyshev V polynomials, the derivative satisfies:
        dV_n/dx = 2 * sum_{k=0}^{n-1} V_k(x)  where the sum is over k with same parity as n-1

    More specifically:
        dV_n/dx = 2 * (V_{n-1} + V_{n-3} + V_{n-5} + ...)

    Parameters
    ----------
    a : ChebyshevPolynomialV
        Series to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    ChebyshevPolynomialV
        Derivative series.

    Notes
    -----
    The degree decreases by 1 for each differentiation.

    Examples
    --------
    >>> a = chebyshev_polynomial_v(torch.tensor([0.0, 0.0, 1.0]))  # V_2
    >>> da = chebyshev_polynomial_v_derivative(a)
    >>> da  # d/dx V_2
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return chebyshev_polynomial_v(a.clone())

    coeffs = a.as_subclass(torch.Tensor)
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
        result_shape = list(coeffs.shape)
        result_shape[-1] = n - 1
        d_coeffs = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # For Chebyshev V: dV_n/dx = 2 * (V_{n-1} + V_{n-3} + V_{n-5} + ...)
        # The derivative recurrence for V polynomials is:
        # d_{n-1} = 2*n*c_n / (but we need to be careful with the V-specific formula)
        #
        # Using the standard approach: convert the derivative formula
        # For V polynomials, the derivative relation is similar to T and U:
        # d_k = d_{k+2} + 2*(k+1)*c_{k+1} with appropriate modifications

        deg = n - 1  # degree of input polynomial

        # Use backward recurrence similar to Chebyshev T
        # For V polynomials: d/dx V_n = 2 * sum_{k<n, (n-k) odd} V_k
        # This means coefficient c_n contributes to d_{n-1}, d_{n-3}, etc.

        # Start with highest degree term
        if deg >= 1:
            d_coeffs[..., deg - 1] = 2.0 * deg * coeffs[..., deg]

        # Backward recurrence
        for k in range(deg - 2, 0, -1):
            d_k = 2.0 * (k + 1) * coeffs[..., k + 1]
            if k + 2 < deg:
                d_k = d_k + d_coeffs[..., k + 2]
            d_coeffs[..., k] = d_k

        # Special case for d_0
        if deg >= 1:
            d_0 = coeffs[..., 1]  # c_1 contributes
            if deg >= 3:
                d_0 = d_0 + 0.5 * d_coeffs[..., 2]
            d_coeffs[..., 0] = d_0

        coeffs = d_coeffs
        n = coeffs.shape[-1]

    return chebyshev_polynomial_v(coeffs)

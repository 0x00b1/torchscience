import torch

from ._chebyshev_polynomial_t import ChebyshevPolynomialT


def chebyshev_polynomial_t_derivative(
    a: ChebyshevPolynomialT,
    order: int = 1,
) -> ChebyshevPolynomialT:
    """Compute derivative of Chebyshev series.

    Uses the recurrence relation for Chebyshev derivatives:
        d_{n-1} = 2*n*c_n
        d_k = d_{k+2} + 2*(k+1)*c_{k+1}  for k = n-2, ..., 1
        d_0 = 0.5*d_2 + c_1

    Parameters
    ----------
    a : ChebyshevPolynomialT
        Series to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    ChebyshevPolynomialT
        Derivative series.

    Notes
    -----
    The degree decreases by 1 for each differentiation.

    Examples
    --------
    >>> a = chebyshev_polynomial_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
    >>> da = chebyshev_polynomial_t_derivative(a)
    >>> da.coeffs  # d/dx T_2 = 4*T_1
    tensor([0., 4.])
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return ChebyshevPolynomialT(coeffs=a.coeffs.clone())

    coeffs = a.coeffs
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

        # Use backward recurrence for numerical stability
        # d_{n-1} = 2*n*c_n (but we index from 0, so d_{n-2} = 2*(n-1)*c_{n-1})
        # Wait, let's be careful about indexing:
        # If input has degree n (n+1 coefficients: c_0, ..., c_n)
        # Output has degree n-1 (n coefficients: d_0, ..., d_{n-1})

        # The recurrence in NumPy's chebder is:
        # d_{n-1} = 2*n*c_n
        # d_k = d_{k+2} + 2*(k+1)*c_{k+1}  for k going down

        deg = n - 1  # degree of input polynomial

        # Start with d_{deg-1} = 2*deg*c_{deg}
        if deg >= 1:
            d_coeffs[..., deg - 1] = 2.0 * deg * coeffs[..., deg]

        # Backward recurrence
        for k in range(deg - 2, 0, -1):
            # d_k = d_{k+2} + 2*(k+1)*c_{k+1}
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

    return ChebyshevPolynomialT(coeffs=coeffs)

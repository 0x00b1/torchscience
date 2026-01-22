import torch

from ._chebyshev_polynomial_u import (
    ChebyshevPolynomialU,
    chebyshev_polynomial_u,
)


def chebyshev_polynomial_u_derivative(
    a: ChebyshevPolynomialU,
    order: int = 1,
) -> ChebyshevPolynomialU:
    """Compute derivative of Chebyshev U series.

    Uses the recurrence relation for Chebyshev U derivatives.
    The derivative of U_n is expressed in terms of U polynomials using
    the relationship:

        d/dx U_n(x) = (n+1) * sum_{k=0, step 2}^{n-1} 2*U_k(x) / (k+1 factor)

    A simpler formulation using backward recurrence is used.

    Parameters
    ----------
    a : ChebyshevPolynomialU
        Series to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    ChebyshevPolynomialU
        Derivative series.

    Notes
    -----
    The degree decreases by 1 for each differentiation.

    Examples
    --------
    >>> a = chebyshev_polynomial_u(torch.tensor([0.0, 0.0, 1.0]))  # U_2
    >>> da = chebyshev_polynomial_u_derivative(a)
    >>> da  # d/dx U_2 = 4*U_1
    ChebyshevPolynomialU(tensor([0., 4.]))
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return chebyshev_polynomial_u(a.clone())

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

        # For U polynomials, derivative of U_k gives contribution to lower U polynomials
        # d/dx U_n(x) in U-basis has specific form
        # Using: d/dx [sum c_k U_k] = sum d_k U_k where
        # d_k comes from summing contributions from c_{k+1}, c_{k+3}, ...
        #
        # The relationship: d/dx U_n = 2(n)*U_{n-1} + 2(n-2)*U_{n-3} + 2(n-4)*U_{n-5} + ...
        # stopping at U_0 or U_1

        result_shape = list(coeffs.shape)
        result_shape[-1] = n - 1
        d_coeffs = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # d/dx U_k contributes 2*k to coefficient of U_{k-1}, plus
        # contributions from higher terms via the recurrence pattern
        # Actually: d/dx U_k(x) = sum_{j=0}^{k-1, same parity} 2*(k-j)*U_j
        # where same parity means j and k-1 have same parity

        for k in range(1, n):
            # Coefficient c_k of U_k contributes to derivative
            c_k = coeffs[..., k]
            # d/dx U_k = 2*k*U_{k-1} + 2*(k-2)*U_{k-3} + 2*(k-4)*U_{k-5} + ...
            for j in range(k - 1, -1, -2):
                factor = 2.0 * (k - (k - 1 - j))
                d_coeffs[..., j] = d_coeffs[..., j] + c_k * factor

        coeffs = d_coeffs
        n = coeffs.shape[-1]

    return chebyshev_polynomial_u(coeffs)

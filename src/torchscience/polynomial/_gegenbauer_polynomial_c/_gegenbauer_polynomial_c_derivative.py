from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_derivative(
    a: GegenbauerPolynomialC,
    order: int = 1,
) -> GegenbauerPolynomialC:
    """Compute derivative of Gegenbauer series.

    Uses the recurrence relation for derivatives of Gegenbauer polynomials:
        d/dx C_n^{lambda}(x) = 2*lambda * C_{n-1}^{lambda+1}(x)

    For derivatives in the same basis (keeping lambda fixed):
        d/dx C_n^{lambda}(x) = 2*lambda * sum_{k=n-1,n-3,...} ((n-k) terms) C_k^{lambda}(x)

    This implementation uses the power basis conversion for accuracy.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        Series to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    GegenbauerPolynomialC
        Derivative series.

    Notes
    -----
    The degree decreases by 1 for each differentiation.

    The derivative of a Gegenbauer polynomial C_n^{lambda} in terms of the
    same basis is:

        d/dx C_n^{lambda}(x) = 2*lambda * C_{n-1}^{lambda+1}(x)

    But to stay in the same basis, we use the identity:

        d/dx C_n^{lambda}(x) = 2*lambda * sum_{j} a_j C_j^{lambda}(x)

    where j = n-1, n-3, ... and a_j can be computed from the connection coefficients.

    This implementation converts to power basis, differentiates, and converts back.

    Examples
    --------
    >>> a = gegenbauer_polynomial_c(torch.tensor([0.0, 1.0]), torch.tensor(1.0))  # C_1^1
    >>> da = gegenbauer_polynomial_c_derivative(a)
    >>> da.coeffs  # d/dx C_1^1 = d/dx (2x) = 2 = 2*C_0^1
    tensor([2.])
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return GegenbauerPolynomialC(
            coeffs=a.coeffs.clone(), lambda_=a.lambda_
        )

    # Convert to power basis, differentiate, convert back
    from torchscience.polynomial._polynomial import polynomial_derivative

    from ._gegenbauer_polynomial_c_to_polynomial import (
        gegenbauer_polynomial_c_to_polynomial,
    )
    from ._polynomial_to_gegenbauer_polynomial_c import (
        polynomial_to_gegenbauer_polynomial_c,
    )

    poly = gegenbauer_polynomial_c_to_polynomial(a)

    for _ in range(order):
        poly = polynomial_derivative(poly)

    return polynomial_to_gegenbauer_polynomial_c(poly, a.lambda_)

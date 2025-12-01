from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_antiderivative(
    a: GegenbauerPolynomialC,
    order: int = 1,
    constant: float = 0.0,
) -> GegenbauerPolynomialC:
    """Compute antiderivative of Gegenbauer series.

    The constant of integration is chosen such that the antiderivative
    evaluates to `constant` at x=0.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        Series to integrate.
    order : int, optional
        Order of integration. Default is 1.
    constant : float, optional
        Integration constant. The antiderivative will evaluate to this
        value at x=0. Default is 0.0.

    Returns
    -------
    GegenbauerPolynomialC
        Antiderivative series.

    Notes
    -----
    The degree increases by 1 for each integration.

    This implementation converts to power basis, integrates, and converts back.

    Examples
    --------
    >>> a = gegenbauer_polynomial_c(torch.tensor([1.0]), torch.tensor(1.0))  # C_0^1 = 1
    >>> ia = gegenbauer_polynomial_c_antiderivative(a)
    >>> ia.coeffs  # integral(1) = x = (1/2)*C_1^1
    tensor([0., 0.5])
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return GegenbauerPolynomialC(
            coeffs=a.coeffs.clone(), lambda_=a.lambda_
        )

    # Convert to power basis, integrate, convert back
    from torchscience.polynomial._polynomial import polynomial_antiderivative

    from ._gegenbauer_polynomial_c_to_polynomial import (
        gegenbauer_polynomial_c_to_polynomial,
    )
    from ._polynomial_to_gegenbauer_polynomial_c import (
        polynomial_to_gegenbauer_polynomial_c,
    )

    poly = gegenbauer_polynomial_c_to_polynomial(a)

    for i in range(order):
        k_val = constant if i == 0 else 0.0
        poly = polynomial_antiderivative(poly, constant=k_val)

    return polynomial_to_gegenbauer_polynomial_c(poly, a.lambda_)

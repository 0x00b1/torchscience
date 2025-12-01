import torch

from torchscience.polynomial._exceptions import ParameterMismatchError

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_multiply(
    a: GegenbauerPolynomialC,
    b: GegenbauerPolynomialC,
) -> GegenbauerPolynomialC:
    """Multiply two Gegenbauer series.

    Uses the linearization formula for Gegenbauer polynomials.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        First series with coefficients a_0, a_1, ..., a_m.
    b : GegenbauerPolynomialC
        Second series with coefficients b_0, b_1, ..., b_n.

    Returns
    -------
    GegenbauerPolynomialC
        Product series with degree at most m + n.

    Raises
    ------
    ParameterMismatchError
        If the series have different lambda parameters.

    Notes
    -----
    The product of two Gegenbauer series of degrees m and n has degree m + n.
    The linearization identity for Gegenbauer polynomials is:

        C_m^{lambda}(x) * C_n^{lambda}(x) = sum_k c_k C_k^{lambda}(x)

    The linearization coefficients involve gamma functions:

        C_m^{lambda}(x) * C_n^{lambda}(x) = sum_{k=0}^{min(m,n)} A_{m,n,k} * C_{m+n-2k}^{lambda}(x)

    where A_{m,n,k} is computed from factorials and lambda.

    This implementation converts to power basis, multiplies, and converts back.

    Examples
    --------
    >>> a = gegenbauer_polynomial_c(torch.tensor([0.0, 1.0]), torch.tensor(1.0))  # C_1^1
    >>> b = gegenbauer_polynomial_c(torch.tensor([0.0, 1.0]), torch.tensor(1.0))  # C_1^1
    >>> c = gegenbauer_polynomial_c_multiply(a, b)
    """
    # Check parameter compatibility
    if not torch.allclose(a.lambda_, b.lambda_):
        raise ParameterMismatchError(
            f"Cannot multiply GegenbauerPolynomialC with lambda={a.lambda_} "
            f"by GegenbauerPolynomialC with lambda={b.lambda_}"
        )

    # Convert to power basis, multiply, convert back
    from torchscience.polynomial._polynomial import polynomial_multiply

    from ._gegenbauer_polynomial_c_to_polynomial import (
        gegenbauer_polynomial_c_to_polynomial,
    )
    from ._polynomial_to_gegenbauer_polynomial_c import (
        polynomial_to_gegenbauer_polynomial_c,
    )

    a_poly = gegenbauer_polynomial_c_to_polynomial(a)
    b_poly = gegenbauer_polynomial_c_to_polynomial(b)

    result_poly = polynomial_multiply(a_poly, b_poly)

    return polynomial_to_gegenbauer_polynomial_c(result_poly, a.lambda_)

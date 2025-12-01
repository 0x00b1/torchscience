from typing import Tuple

from torchscience.polynomial._exceptions import ParameterMismatchError

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_divmod(
    a: GegenbauerPolynomialC,
    b: GegenbauerPolynomialC,
) -> Tuple[GegenbauerPolynomialC, GegenbauerPolynomialC]:
    """Divide two Gegenbauer series with remainder.

    Returns quotient q and remainder r such that a = b*q + r.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        Dividend.
    b : GegenbauerPolynomialC
        Divisor.

    Returns
    -------
    Tuple[GegenbauerPolynomialC, GegenbauerPolynomialC]
        (quotient, remainder)

    Raises
    ------
    ParameterMismatchError
        If the series have different lambda parameters.

    Notes
    -----
    Performs polynomial division by converting to power basis,
    dividing, and converting back.

    Examples
    --------
    >>> a = gegenbauer_polynomial_c(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(1.0))
    >>> b = gegenbauer_polynomial_c(torch.tensor([1.0, 1.0]), torch.tensor(1.0))
    >>> q, r = gegenbauer_polynomial_c_divmod(a, b)
    """
    import torch

    # Check parameter compatibility
    if not torch.allclose(a.lambda_, b.lambda_):
        raise ParameterMismatchError(
            f"Cannot divide GegenbauerPolynomialC with lambda={a.lambda_} "
            f"by GegenbauerPolynomialC with lambda={b.lambda_}"
        )

    # Convert to power basis, divide, convert back
    from torchscience.polynomial._polynomial import polynomial_divmod

    from ._gegenbauer_polynomial_c_to_polynomial import (
        gegenbauer_polynomial_c_to_polynomial,
    )
    from ._polynomial_to_gegenbauer_polynomial_c import (
        polynomial_to_gegenbauer_polynomial_c,
    )

    a_poly = gegenbauer_polynomial_c_to_polynomial(a)
    b_poly = gegenbauer_polynomial_c_to_polynomial(b)

    q_poly, r_poly = polynomial_divmod(a_poly, b_poly)

    q = polynomial_to_gegenbauer_polynomial_c(q_poly, a.lambda_)
    r = polynomial_to_gegenbauer_polynomial_c(r_poly, a.lambda_)

    return q, r

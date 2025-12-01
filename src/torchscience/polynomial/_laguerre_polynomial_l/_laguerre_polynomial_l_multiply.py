import numpy as np
import torch

from ._laguerre_polynomial_l import LaguerrePolynomialL


def laguerre_polynomial_l_multiply(
    a: LaguerrePolynomialL,
    b: LaguerrePolynomialL,
) -> LaguerrePolynomialL:
    """Multiply two Laguerre series.

    Uses NumPy's Laguerre linearization for the product.

    Parameters
    ----------
    a : LaguerrePolynomialL
        First series with coefficients a_0, a_1, ..., a_m.
    b : LaguerrePolynomialL
        Second series with coefficients b_0, b_1, ..., b_n.

    Returns
    -------
    LaguerrePolynomialL
        Product series with degree at most m + n.

    Notes
    -----
    The product of two Laguerre series of degrees m and n has degree m + n.
    The linearization identity for Laguerre polynomials ensures the product
    remains in Laguerre form:

        L_m(x) * L_n(x) = sum_k c_k L_k(x)

    where the coefficients c_k are determined by the linearization formula.

    Examples
    --------
    >>> a = laguerre_polynomial_l(torch.tensor([0.0, 1.0]))  # L_1
    >>> b = laguerre_polynomial_l(torch.tensor([0.0, 1.0]))  # L_1
    >>> c = laguerre_polynomial_l_multiply(a, b)
    >>> c.coeffs  # L_1 * L_1
    tensor([1., -4.,  2.])
    """
    a_coeffs = a.coeffs
    b_coeffs = b.coeffs

    # Use NumPy's lagmul which implements the linearization formula
    a_np = a_coeffs.detach().cpu().numpy()
    b_np = b_coeffs.detach().cpu().numpy()

    result_np = np.polynomial.laguerre.lagmul(a_np, b_np)

    result_coeffs = torch.from_numpy(result_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )

    return LaguerrePolynomialL(coeffs=result_coeffs)

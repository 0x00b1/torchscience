import numpy as np
import torch

from ._legendre_polynomial_p import (
    LegendrePolynomialP,
    legendre_polynomial_p,
)


def legendre_polynomial_p_multiply(
    a: LegendrePolynomialP,
    b: LegendrePolynomialP,
) -> LegendrePolynomialP:
    """Multiply two Legendre series.

    Uses NumPy's Legendre linearization for the product.

    Parameters
    ----------
    a : LegendrePolynomialP
        First series with coefficients a_0, a_1, ..., a_m.
    b : LegendrePolynomialP
        Second series with coefficients b_0, b_1, ..., b_n.

    Returns
    -------
    LegendrePolynomialP
        Product series with degree at most m + n.

    Notes
    -----
    The product of two Legendre series of degrees m and n has degree m + n.
    The linearization identity for Legendre polynomials ensures the product
    remains in Legendre form:

        P_m(x) * P_n(x) = sum_k c_k P_k(x)

    where the coefficients c_k are determined by the linearization formula.

    Examples
    --------
    >>> a = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
    >>> b = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
    >>> c = legendre_polynomial_p_multiply(a, b)
    >>> c  # P_1 * P_1 = (1/3)*P_0 + (2/3)*P_2
    LegendrePolynomialP(tensor([0.3333, 0.0000, 0.6667]))
    """
    # Convert to plain tensors to avoid operator interception
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    # Use NumPy's legmul which implements the linearization formula
    a_np = a_coeffs.detach().cpu().numpy()
    b_np = b_coeffs.detach().cpu().numpy()

    result_np = np.polynomial.legendre.legmul(a_np, b_np)

    result_coeffs = torch.from_numpy(result_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )

    return legendre_polynomial_p(result_coeffs)

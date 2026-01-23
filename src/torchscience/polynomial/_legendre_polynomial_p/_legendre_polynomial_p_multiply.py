import torch

from ._legendre_polynomial_p import (
    LegendrePolynomialP,
)
from ._legendre_polynomial_p_to_polynomial import (
    legendre_polynomial_p_to_polynomial,
)
from ._polynomial_to_legendre_polynomial_p import (
    polynomial_to_legendre_polynomial_p,
)


def _polynomial_convolve(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Multiply two polynomial coefficient tensors (convolution).

    Pure PyTorch implementation of polynomial multiplication.
    """
    n_p = p.shape[-1]
    n_q = q.shape[-1]
    n_out = n_p + n_q - 1

    result = torch.zeros(n_out, dtype=p.dtype, device=p.device)

    for i in range(n_p):
        for j in range(n_q):
            result[i + j] = result[i + j] + p[i] * q[j]

    return result


def legendre_polynomial_p_multiply(
    a: LegendrePolynomialP,
    b: LegendrePolynomialP,
) -> LegendrePolynomialP:
    """Multiply two Legendre series.

    Converts to polynomial basis, multiplies, and converts back.

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
    The linearization identity ensures the product remains in Legendre form.

    This implementation is pure PyTorch and supports autograd, GPU tensors,
    and torch.compile.

    Examples
    --------
    >>> a = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
    >>> b = legendre_polynomial_p(torch.tensor([0.0, 1.0]))  # P_1
    >>> c = legendre_polynomial_p_multiply(a, b)
    >>> c  # P_1 * P_1 = (1/3)*P_0 + (2/3)*P_2
    LegendrePolynomialP(tensor([0.3333, 0.0000, 0.6667]))
    """
    # Convert to polynomial basis
    p_a = legendre_polynomial_p_to_polynomial(a)
    p_b = legendre_polynomial_p_to_polynomial(b)

    # Get coefficient tensors
    p_a_coeffs = p_a.as_subclass(torch.Tensor)
    p_b_coeffs = p_b.as_subclass(torch.Tensor)

    # Multiply in polynomial basis (convolution)
    from torchscience.polynomial._polynomial import Polynomial

    p_c_coeffs = _polynomial_convolve(p_a_coeffs, p_b_coeffs)
    p_c = Polynomial(p_c_coeffs)

    # Convert back to Legendre basis
    return polynomial_to_legendre_polynomial_p(p_c)

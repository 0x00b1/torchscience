import torch

from ._laguerre_polynomial_l import (
    LaguerrePolynomialL,
)
from ._laguerre_polynomial_l_to_polynomial import (
    laguerre_polynomial_l_to_polynomial,
)
from ._polynomial_to_laguerre_polynomial_l import (
    polynomial_to_laguerre_polynomial_l,
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


def laguerre_polynomial_l_multiply(
    a: LaguerrePolynomialL,
    b: LaguerrePolynomialL,
) -> LaguerrePolynomialL:
    """Multiply two Laguerre series.

    Converts to polynomial basis, multiplies, and converts back.

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
    The linearization identity ensures the product remains in Laguerre form.

    This implementation is pure PyTorch and supports autograd, GPU tensors,
    and torch.compile.

    Examples
    --------
    >>> a = laguerre_polynomial_l(torch.tensor([0.0, 1.0]))  # L_1
    >>> b = laguerre_polynomial_l(torch.tensor([0.0, 1.0]))  # L_1
    >>> c = laguerre_polynomial_l_multiply(a, b)
    >>> c  # L_1 * L_1 = L_0 - 2*L_1 + 2*L_2
    LaguerrePolynomialL(tensor([1., -2.,  2.]))
    """
    # Convert to polynomial basis
    p_a = laguerre_polynomial_l_to_polynomial(a)
    p_b = laguerre_polynomial_l_to_polynomial(b)

    # Get coefficient tensors
    p_a_coeffs = p_a.as_subclass(torch.Tensor)
    p_b_coeffs = p_b.as_subclass(torch.Tensor)

    # Multiply in polynomial basis (convolution)
    from torchscience.polynomial._polynomial import Polynomial

    p_c_coeffs = _polynomial_convolve(p_a_coeffs, p_b_coeffs)
    p_c = Polynomial(p_c_coeffs)

    # Convert back to Laguerre basis
    return polynomial_to_laguerre_polynomial_l(p_c)

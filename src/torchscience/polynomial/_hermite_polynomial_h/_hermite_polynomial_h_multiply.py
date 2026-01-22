import numpy as np
import torch

from ._hermite_polynomial_h import (
    HermitePolynomialH,
    hermite_polynomial_h,
)


def hermite_polynomial_h_multiply(
    a: HermitePolynomialH,
    b: HermitePolynomialH,
) -> HermitePolynomialH:
    """Multiply two Physicists' Hermite series.

    Uses NumPy's Hermite linearization for the product.

    Parameters
    ----------
    a : HermitePolynomialH
        First series with coefficients a_0, a_1, ..., a_m.
    b : HermitePolynomialH
        Second series with coefficients b_0, b_1, ..., b_n.

    Returns
    -------
    HermitePolynomialH
        Product series with degree at most m + n.

    Notes
    -----
    The product of two Hermite series of degrees m and n has degree m + n.
    The linearization identity for Hermite polynomials ensures the product
    remains in Hermite form:

        H_m(x) * H_n(x) = sum_k c_k H_k(x)

    where the coefficients c_k are determined by the linearization formula.

    Examples
    --------
    >>> a = hermite_polynomial_h(torch.tensor([0.0, 1.0]))  # H_1
    >>> b = hermite_polynomial_h(torch.tensor([0.0, 1.0]))  # H_1
    >>> c = hermite_polynomial_h_multiply(a, b)
    >>> # H_1 * H_1 = H_2 + 2 = (4x^2 - 2) + 2 = 4x^2 which is H_2 + 2*H_0
    """
    # Convert to plain tensors to avoid operator interception
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    # Use NumPy's hermmul which implements the linearization formula
    # Note: numpy.polynomial.hermite uses the "physicists'" convention
    a_np = a_coeffs.detach().cpu().numpy()
    b_np = b_coeffs.detach().cpu().numpy()

    result_np = np.polynomial.hermite.hermmul(a_np, b_np)

    result_coeffs = torch.from_numpy(result_np).to(
        dtype=a_coeffs.dtype, device=a_coeffs.device
    )

    return hermite_polynomial_h(result_coeffs)

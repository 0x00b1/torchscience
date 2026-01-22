import torch

from ._chebyshev_polynomial_u import (
    ChebyshevPolynomialU,
    chebyshev_polynomial_u,
)


def chebyshev_polynomial_u_multiply(
    a: ChebyshevPolynomialU,
    b: ChebyshevPolynomialU,
) -> ChebyshevPolynomialU:
    """Multiply two Chebyshev U series.

    Uses the linearization formula for Chebyshev polynomials of the second kind:
        U_m(x) * U_n(x) = sum_{k=0}^{min(m,n)} U_{m+n-2k}(x)

    Parameters
    ----------
    a : ChebyshevPolynomialU
        First series with coefficients a_0, a_1, ..., a_m.
    b : ChebyshevPolynomialU
        Second series with coefficients b_0, b_1, ..., b_n.

    Returns
    -------
    ChebyshevPolynomialU
        Product series with degree at most m + n.

    Notes
    -----
    The product of two Chebyshev U series of degrees m and n has degree m + n.
    The linearization formula ensures the product remains in Chebyshev U form.

    Examples
    --------
    >>> a = chebyshev_polynomial_u(torch.tensor([0.0, 1.0]))  # U_1
    >>> b = chebyshev_polynomial_u(torch.tensor([0.0, 1.0]))  # U_1
    >>> c = chebyshev_polynomial_u_multiply(a, b)
    >>> c  # U_1 * U_1 = U_0 + U_2
    ChebyshevPolynomialU(tensor([1., 0., 1.]))
    """
    # Convert to plain tensors to avoid operator interception
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    # Result has degree (n_a - 1) + (n_b - 1) = n_a + n_b - 2
    # So we need n_a + n_b - 1 coefficients
    n_c = n_a + n_b - 1

    # Initialize result coefficients
    result_shape = list(a_coeffs.shape)
    result_shape[-1] = n_c
    c_coeffs = torch.zeros(
        result_shape, dtype=a_coeffs.dtype, device=a_coeffs.device
    )

    # Apply linearization: U_i * U_j = sum_{k=0}^{min(i,j)} U_{i+j-2k}
    # This is because U_m * U_n = U_{m+n} + U_{m+n-2} + ... + U_{|m-n|}
    for i in range(n_a):
        for j in range(n_b):
            # Contribution: a_i * b_j * (U_{i+j} + U_{i+j-2} + ... + U_{|i-j|})
            coeff_product = a_coeffs[..., i] * b_coeffs[..., j]

            # Sum over k from 0 to min(i, j)
            min_ij = min(i, j)
            for k in range(min_ij + 1):
                idx = i + j - 2 * k
                c_coeffs[..., idx] = c_coeffs[..., idx] + coeff_product

    return chebyshev_polynomial_u(c_coeffs)

import torch

from ._chebyshev_polynomial_v import ChebyshevPolynomialV


def chebyshev_polynomial_v_multiply(
    a: ChebyshevPolynomialV,
    b: ChebyshevPolynomialV,
) -> ChebyshevPolynomialV:
    """Multiply two Chebyshev V series.

    Uses the linearization formula for Chebyshev V polynomials:
        V_m(x) * V_n(x) = 0.5 * (V_{m+n}(x) + V_{|m-n|}(x))

    Parameters
    ----------
    a : ChebyshevPolynomialV
        First series with coefficients a_0, a_1, ..., a_m.
    b : ChebyshevPolynomialV
        Second series with coefficients b_0, b_1, ..., b_n.

    Returns
    -------
    ChebyshevPolynomialV
        Product series with degree at most m + n.

    Notes
    -----
    The product of two Chebyshev V series of degrees m and n has degree m + n.
    The linearization formula ensures the product remains in Chebyshev V form.

    The Chebyshev V polynomials satisfy:
        V_m(x) * V_n(x) = 0.5 * (V_{m+n}(x) + V_{|m-n|}(x))

    Examples
    --------
    >>> a = chebyshev_polynomial_v(torch.tensor([0.0, 1.0]))  # V_1
    >>> b = chebyshev_polynomial_v(torch.tensor([0.0, 1.0]))  # V_1
    >>> c = chebyshev_polynomial_v_multiply(a, b)
    >>> c.coeffs  # V_1 * V_1 = 0.5*(V_0 + V_2)
    tensor([0.5, 0.0, 0.5])
    """
    a_coeffs = a.coeffs
    b_coeffs = b.coeffs

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

    # Apply linearization: V_i * V_j = 0.5 * (V_{i+j} + V_{|i-j|})
    for i in range(n_a):
        for j in range(n_b):
            # Contribution: a_i * b_j * (0.5 * V_{i+j} + 0.5 * V_{|i-j|})
            coeff_product = a_coeffs[..., i] * b_coeffs[..., j]

            idx_sum = i + j
            idx_diff = abs(i - j)

            if i == 0 or j == 0:
                # V_0 * V_k = V_k (no factor of 0.5)
                c_coeffs[..., idx_sum] = c_coeffs[..., idx_sum] + coeff_product
            else:
                # V_i * V_j = 0.5 * (V_{i+j} + V_{|i-j|})
                c_coeffs[..., idx_sum] = (
                    c_coeffs[..., idx_sum] + 0.5 * coeff_product
                )
                c_coeffs[..., idx_diff] = (
                    c_coeffs[..., idx_diff] + 0.5 * coeff_product
                )

    return ChebyshevPolynomialV(coeffs=c_coeffs)

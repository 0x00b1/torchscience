import torch

from ._chebyshev_polynomial_w import (
    ChebyshevPolynomialW,
    chebyshev_polynomial_w,
)


def chebyshev_polynomial_w_multiply(
    a: ChebyshevPolynomialW,
    b: ChebyshevPolynomialW,
) -> ChebyshevPolynomialW:
    """Multiply two Chebyshev W series.

    Uses the linearization formula for Chebyshev W polynomials:
        W_m(x) * W_n(x) = 0.5 * (W_{m+n}(x) + W_{|m-n|}(x))

    Parameters
    ----------
    a : ChebyshevPolynomialW
        First series with coefficients a_0, a_1, ..., a_m.
    b : ChebyshevPolynomialW
        Second series with coefficients b_0, b_1, ..., b_n.

    Returns
    -------
    ChebyshevPolynomialW
        Product series with degree at most m + n.

    Notes
    -----
    The product of two Chebyshev W series of degrees m and n has degree m + n.
    The linearization formula ensures the product remains in Chebyshev W form.

    The Chebyshev W polynomials satisfy:
        W_m(x) * W_n(x) = 0.5 * (W_{m+n}(x) + W_{|m-n|}(x))

    Examples
    --------
    >>> a = chebyshev_polynomial_w(torch.tensor([0.0, 1.0]))  # W_1
    >>> b = chebyshev_polynomial_w(torch.tensor([0.0, 1.0]))  # W_1
    >>> c = chebyshev_polynomial_w_multiply(a, b)
    >>> c  # W_1 * W_1 = 0.5*(W_0 + W_2)
    ChebyshevPolynomialW(tensor([0.5, 0.0, 0.5]))
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

    # Apply linearization: W_i * W_j = 0.5 * (W_{i+j} + W_{|i-j|})
    for i in range(n_a):
        for j in range(n_b):
            # Contribution: a_i * b_j * (0.5 * W_{i+j} + 0.5 * W_{|i-j|})
            coeff_product = a_coeffs[..., i] * b_coeffs[..., j]

            idx_sum = i + j
            idx_diff = abs(i - j)

            if i == 0 or j == 0:
                # W_0 * W_k = W_k (no factor of 0.5)
                c_coeffs[..., idx_sum] = c_coeffs[..., idx_sum] + coeff_product
            else:
                # W_i * W_j = 0.5 * (W_{i+j} + W_{|i-j|})
                c_coeffs[..., idx_sum] = (
                    c_coeffs[..., idx_sum] + 0.5 * coeff_product
                )
                c_coeffs[..., idx_diff] = (
                    c_coeffs[..., idx_diff] + 0.5 * coeff_product
                )

    return chebyshev_polynomial_w(c_coeffs)

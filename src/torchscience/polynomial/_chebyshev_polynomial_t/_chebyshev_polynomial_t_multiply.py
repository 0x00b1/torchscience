import torch

from ._chebyshev_polynomial_t import (
    ChebyshevPolynomialT,
    chebyshev_polynomial_t,
)


def chebyshev_polynomial_t_multiply(
    a: ChebyshevPolynomialT,
    b: ChebyshevPolynomialT,
) -> ChebyshevPolynomialT:
    """Multiply two Chebyshev series.

    Uses the linearization formula:
        T_m(x) * T_n(x) = 0.5 * (T_{m+n}(x) + T_{|m-n|}(x))

    Parameters
    ----------
    a : ChebyshevPolynomialT
        First series with coefficients a_0, a_1, ..., a_m.
    b : ChebyshevPolynomialT
        Second series with coefficients b_0, b_1, ..., b_n.

    Returns
    -------
    ChebyshevPolynomialT
        Product series with degree at most m + n.

    Notes
    -----
    The product of two Chebyshev series of degrees m and n has degree m + n.
    The linearization formula ensures the product remains in Chebyshev form.

    Examples
    --------
    >>> a = chebyshev_polynomial_t(torch.tensor([0.0, 1.0]))  # T_1
    >>> b = chebyshev_polynomial_t(torch.tensor([0.0, 1.0]))  # T_1
    >>> c = chebyshev_polynomial_t_multiply(a, b)
    >>> c  # T_1 * T_1 = 0.5*(T_0 + T_2)
    ChebyshevPolynomialT(tensor([0.5, 0.0, 0.5]))
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

    # Apply linearization: T_i * T_j = 0.5 * (T_{i+j} + T_{|i-j|})
    for i in range(n_a):
        for j in range(n_b):
            # Contribution: a_i * b_j * (0.5 * T_{i+j} + 0.5 * T_{|i-j|})
            coeff_product = a_coeffs[..., i] * b_coeffs[..., j]

            idx_sum = i + j
            idx_diff = abs(i - j)

            if i == 0 or j == 0:
                # T_0 * T_k = T_k (no factor of 0.5)
                c_coeffs[..., idx_sum] = c_coeffs[..., idx_sum] + coeff_product
            else:
                # T_i * T_j = 0.5 * (T_{i+j} + T_{|i-j|})
                c_coeffs[..., idx_sum] = (
                    c_coeffs[..., idx_sum] + 0.5 * coeff_product
                )
                c_coeffs[..., idx_diff] = (
                    c_coeffs[..., idx_diff] + 0.5 * coeff_product
                )

    return chebyshev_polynomial_t(c_coeffs)

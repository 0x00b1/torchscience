import math

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

    Uses the linearization formula for Hermite polynomials (physicists' convention):

        H_m(x) * H_n(x) = sum_{k=0}^{min(m,n)} 2^k * k! * C(m,k) * C(n,k) * H_{m+n-2k}(x)

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
    The linearization identity ensures the product remains in Hermite form.

    This implementation is pure PyTorch and supports autograd, GPU tensors,
    and torch.compile.

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

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    # Result has degree (n_a - 1) + (n_b - 1) = n_a + n_b - 2
    # So we need n_a + n_b - 1 coefficients
    n_c = n_a + n_b - 1

    # Precompute 2^k * k! for linearization coefficients
    max_k = min(n_a, n_b)
    two_pow_k_factorial = [(2**k) * math.factorial(k) for k in range(max_k)]

    # Build result by accumulating contributions (for autograd support)
    # Apply linearization: H_i * H_j = sum_{k=0}^{min(i,j)} 2^k * k! * C(i,k) * C(j,k) * H_{i+j-2k}
    contributions = []
    for idx in range(n_c):
        contrib = torch.zeros_like(a_coeffs[..., 0])
        for i in range(n_a):
            for j in range(n_b):
                # Check if (i,j) contributes to index idx
                # idx = i + j - 2*k => k = (i + j - idx) / 2
                diff = i + j - idx
                if diff < 0 or diff % 2 != 0:
                    continue
                k = diff // 2
                if k > min(i, j):
                    continue

                # Linearization coefficient: 2^k * k! * C(i,k) * C(j,k)
                linearization_coeff = (
                    two_pow_k_factorial[k] * math.comb(i, k) * math.comb(j, k)
                )
                contrib = (
                    contrib
                    + a_coeffs[..., i] * b_coeffs[..., j] * linearization_coeff
                )
        contributions.append(contrib)

    c_coeffs = torch.stack(contributions, dim=-1)
    return hermite_polynomial_h(c_coeffs)

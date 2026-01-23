"""Multiply two Probabilists' Hermite series using linearization."""

from __future__ import annotations

import math

import torch

from ._hermite_polynomial_he import (
    HermitePolynomialHe,
    hermite_polynomial_he,
)


def hermite_polynomial_he_multiply(
    a: HermitePolynomialHe,
    b: HermitePolynomialHe,
) -> HermitePolynomialHe:
    """Multiply two Probabilists' Hermite series.

    Uses the linearization formula for Hermite polynomials of the second kind
    (probabilists' convention):

        He_m(x) * He_n(x) = sum_{k=0}^{min(m,n)} k! * C(m,k) * C(n,k) * He_{m+n-2k}(x)

    Parameters
    ----------
    a : HermitePolynomialHe
        First series with coefficients a_0, a_1, ..., a_m.
    b : HermitePolynomialHe
        Second series with coefficients b_0, b_1, ..., b_n.

    Returns
    -------
    HermitePolynomialHe
        Product series with degree at most m + n.

    Notes
    -----
    The product of two Hermite series of degrees m and n has degree m + n.
    The linearization identity ensures the product remains in Hermite form.

    This implementation is pure PyTorch and supports autograd, GPU tensors,
    and torch.compile.

    Examples
    --------
    >>> a = hermite_polynomial_he(torch.tensor([0.0, 1.0]))  # He_1 = x
    >>> b = hermite_polynomial_he(torch.tensor([0.0, 1.0]))  # He_1 = x
    >>> c = hermite_polynomial_he_multiply(a, b)
    >>> # He_1 * He_1 = x^2 = He_2 + 1 = He_2 + He_0
    """
    # Convert to plain tensors to avoid operator interception
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    # Result has degree (n_a - 1) + (n_b - 1) = n_a + n_b - 2
    # So we need n_a + n_b - 1 coefficients
    n_c = n_a + n_b - 1

    # Precompute factorials for linearization coefficients
    max_k = min(n_a, n_b)
    factorials = [math.factorial(k) for k in range(max_k)]

    # Build result by accumulating contributions (for autograd support)
    # Apply linearization: He_i * He_j = sum_{k=0}^{min(i,j)} k! * C(i,k) * C(j,k) * He_{i+j-2k}
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

                # Linearization coefficient: k! * C(i,k) * C(j,k)
                linearization_coeff = (
                    factorials[k] * math.comb(i, k) * math.comb(j, k)
                )
                contrib = (
                    contrib
                    + a_coeffs[..., i] * b_coeffs[..., j] * linearization_coeff
                )
        contributions.append(contrib)

    c_coeffs = torch.stack(contributions, dim=-1)
    return hermite_polynomial_he(c_coeffs)

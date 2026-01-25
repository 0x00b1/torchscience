"""Bilinear transform for analog to digital filter conversion (BA form)."""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def bilinear_transform_ba(
    numerator: Tensor,
    denominator: Tensor,
    sampling_frequency: float,
    prewarp_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Transform analog transfer function to digital using bilinear transform.

    Applies the bilinear (Tustin) transformation s = k*(z-1)/(z+1) to convert
    an analog filter H(s) = B(s)/A(s) to a digital filter H(z) = B(z)/A(z).

    Parameters
    ----------
    numerator : Tensor
        Analog numerator polynomial coefficients in descending powers of s,
        i.e., [b_n, b_{n-1}, ..., b_1, b_0] for B(s) = b_n*s^n + ... + b_0.
    denominator : Tensor
        Analog denominator polynomial coefficients in descending powers of s,
        i.e., [a_m, a_{m-1}, ..., a_1, a_0] for A(s) = a_m*s^m + ... + a_0.
    sampling_frequency : float
        Target sampling frequency in Hz.
    prewarp_frequency : float, optional
        Frequency (in Hz) at which analog and digital responses should match
        exactly. If None (default), no prewarping is applied and the standard
        bilinear transform is used with k = 2*sampling_frequency.
    dtype : torch.dtype, optional
        Output dtype. Defaults to input dtype.
    device : torch.device, optional
        Output device. Defaults to input device.

    Returns
    -------
    numerator_digital : Tensor
        Digital numerator coefficients in descending powers of z.
    denominator_digital : Tensor
        Digital denominator coefficients in descending powers of z.

    Notes
    -----
    The bilinear transform substitutes:

    .. math::
        s = k \\frac{z - 1}{z + 1}

    where k = 2*sampling_frequency for the standard transform, or
    k = prewarp_frequency / tan(prewarp_frequency * pi / sampling_frequency)
    for prewarped transform.

    For most filter design workflows, prefer `bilinear_transform_zpk` which
    is more numerically stable for high-order filters. This BA variant is
    provided for cases where the transfer function polynomial form is already
    available or required.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import bilinear_transform_ba
    >>> # First-order lowpass: H(s) = 1 / (s + 1)
    >>> b = torch.tensor([1.0])
    >>> a = torch.tensor([1.0, 1.0])
    >>> b_d, a_d = bilinear_transform_ba(b, a, sampling_frequency=10.0)
    """
    if dtype is None:
        dtype = numerator.dtype
    if device is None:
        device = numerator.device

    # Convert to float64 for numerical stability
    b = numerator.to(dtype=torch.float64, device=device)
    a = denominator.to(dtype=torch.float64, device=device)

    # Compute bilinear transform coefficient
    if prewarp_frequency is not None:
        # Prewarped: k = w_prewarp / tan(w_prewarp / (2*fs))
        w_prewarp = 2 * math.pi * prewarp_frequency
        k = w_prewarp / math.tan(w_prewarp / (2 * sampling_frequency))
    else:
        # Standard: k = 2 * fs
        k = 2.0 * sampling_frequency

    # Get polynomial degrees
    n = b.numel() - 1  # numerator degree
    m = a.numel() - 1  # denominator degree

    # The final polynomial degree is max(n, m)
    order = max(n, m)

    # Pad polynomials to same length (order + 1)
    if n < order:
        b = torch.cat(
            [torch.zeros(order - n, dtype=b.dtype, device=device), b]
        )
    if m < order:
        a = torch.cat(
            [torch.zeros(order - m, dtype=a.dtype, device=device), a]
        )

    # Transform using the bilinear substitution
    # s = k*(z-1)/(z+1) means we substitute and collect powers of z
    # For a term b_i * s^i, after substitution and multiplying by (z+1)^order:
    # b_i * k^i * (z-1)^i * (z+1)^(order-i)

    b_d = torch.zeros(order + 1, dtype=b.dtype, device=device)
    a_d = torch.zeros(order + 1, dtype=a.dtype, device=device)

    # Precompute binomial coefficients for (z-1)^i and (z+1)^j
    # (z-1)^i = sum_{r=0}^i C(i,r) * z^r * (-1)^(i-r)
    # (z+1)^j = sum_{s=0}^j C(j,s) * z^s

    for i in range(order + 1):
        # Coefficient for s^(order-i) in original polynomial (reversed indexing)
        power = order - i
        b_coeff = b[i]
        a_coeff = a[i]

        # k^power factor
        k_power = k**power

        # (z-1)^power * (z+1)^(order-power)
        # We need to compute this product and add to result

        # First compute (z-1)^power
        z_minus_1 = _binomial_expand(power, -1, device)
        # Then compute (z+1)^(order-power)
        z_plus_1 = _binomial_expand(order - power, 1, device)
        # Multiply polynomials
        product = _poly_multiply(z_minus_1, z_plus_1)

        # Add contribution to numerator and denominator
        b_d = b_d + b_coeff * k_power * product
        a_d = a_d + a_coeff * k_power * product

    # Normalize so that a_d[0] = 1
    norm = a_d[0]
    b_d = b_d / norm
    a_d = a_d / norm

    return b_d.to(dtype), a_d.to(dtype)


def _binomial_expand(n: int, sign: int, device: torch.device) -> Tensor:
    """Expand (z + sign)^n as polynomial coefficients in descending order.

    Parameters
    ----------
    n : int
        Power to expand.
    sign : int
        Either +1 for (z+1)^n or -1 for (z-1)^n.
    device : torch.device
        Device for output tensor.

    Returns
    -------
    coeffs : Tensor
        Polynomial coefficients [z^n, z^{n-1}, ..., z^0].
    """
    coeffs = torch.zeros(n + 1, dtype=torch.float64, device=device)
    for k in range(n + 1):
        # C(n,k) * sign^(n-k) for z^k term
        # In descending order: z^(n-k) coefficient
        binom = _binomial_coeff(n, k)
        coeffs[n - k] = binom * (sign ** (n - k))
    return coeffs


def _binomial_coeff(n: int, k: int) -> float:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    # Use symmetry for efficiency
    k = min(k, n - k)
    result = 1.0
    for i in range(k):
        result = result * (n - i) / (i + 1)
    return result


def _poly_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two polynomials (convolution).

    Parameters
    ----------
    a, b : Tensor
        Polynomial coefficients in descending order.

    Returns
    -------
    result : Tensor
        Product polynomial coefficients.
    """
    n = a.numel()
    m = b.numel()
    result = torch.zeros(n + m - 1, dtype=a.dtype, device=a.device)

    for i in range(n):
        for j in range(m):
            result[i + j] += a[i] * b[j]

    return result

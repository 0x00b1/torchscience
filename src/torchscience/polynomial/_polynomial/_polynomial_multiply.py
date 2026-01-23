"""Polynomial multiplication with adaptive algorithm selection.

Uses direct convolution (C++ kernel) for low-degree polynomials and
FFT-based multiplication for high-degree polynomials.
"""

import torch

from ._polynomial import Polynomial, polynomial

# Threshold for switching between direct and FFT multiplication.
# Below this output size, direct convolution is typically faster.
# At or above this size, FFT's O(n log n) beats O(n^2) direct convolution.
FFT_CROSSOVER_THRESHOLD = 64


def _multiply_direct(p: Polynomial, q: Polynomial) -> Polynomial:
    """Multiply polynomials using direct convolution (C++ kernel).

    This is efficient for low-degree polynomials due to lower overhead.

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to multiply.

    Returns
    -------
    Polynomial
        Product p * q.
    """
    # Get shapes
    p_batch = p.shape[:-1]
    q_batch = q.shape[:-1]
    n_p = p.shape[-1]
    n_q = q.shape[-1]

    # Handle empty polynomials
    if n_p == 0 or n_q == 0:
        broadcast_batch = torch.broadcast_shapes(p_batch, q_batch)
        return polynomial(
            torch.zeros(
                *broadcast_batch,
                1,  # Must have at least one coefficient
                dtype=p.dtype,
                device=p.device,
            )
        )

    # Broadcast batch dimensions
    broadcast_batch = torch.broadcast_shapes(p_batch, q_batch)

    # Expand to broadcast shape
    p_expanded = p.expand(*broadcast_batch, n_p)
    q_expanded = q.expand(*broadcast_batch, n_q)

    # Flatten batch dimensions for kernel: (...batch, N) -> (B, N)
    batch_size = broadcast_batch.numel() if len(broadcast_batch) > 0 else 1
    p_flat = p_expanded.reshape(batch_size, n_p).contiguous()
    q_flat = q_expanded.reshape(batch_size, n_q).contiguous()

    # Promote to common dtype
    common_dtype = torch.promote_types(p_flat.dtype, q_flat.dtype)
    p_flat = p_flat.to(common_dtype)
    q_flat = q_flat.to(common_dtype)

    # Call C++ kernel
    result_flat = torch.ops.torchscience.polynomial_multiply(p_flat, q_flat)

    # Reshape output back to broadcast batch dimensions
    n_out = n_p + n_q - 1
    if len(broadcast_batch) == 0:
        result = result_flat.reshape(n_out)
    else:
        result = result_flat.reshape(*broadcast_batch, n_out)

    return polynomial(result)


def _multiply_fft(p: Polynomial, q: Polynomial) -> Polynomial:
    """Multiply polynomials using FFT-based convolution.

    This is efficient for high-degree polynomials due to O(n log n) complexity.

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to multiply.

    Returns
    -------
    Polynomial
        Product p * q.
    """
    from ._polynomial_multiply_fft import polynomial_multiply_fft

    return polynomial_multiply_fft(p, q)


def polynomial_multiply(p: Polynomial, q: Polynomial) -> Polynomial:
    """Multiply two polynomials with adaptive algorithm selection.

    Automatically chooses between direct convolution (C++ kernel) for
    low-degree polynomials and FFT-based multiplication for high-degree
    polynomials. The crossover threshold is at output size 64.

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to multiply.

    Returns
    -------
    Polynomial
        Product p * q with degree = deg(p) + deg(q).

    Notes
    -----
    - For output size < 64: Uses direct O(n^2) convolution via C++ kernel
    - For output size >= 64: Uses FFT-based O(n log n) multiplication

    The threshold of 64 is chosen based on the crossover point where
    FFT overhead is amortized by faster asymptotic complexity.

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> q = polynomial(torch.tensor([4.0, 5.0]))       # 4 + 5x
    >>> r = polynomial_multiply(p, q)                   # 4 + 13x + 22x^2 + 15x^3
    """
    n_p = p.shape[-1]
    n_q = q.shape[-1]
    n_out = n_p + n_q - 1

    if n_out < FFT_CROSSOVER_THRESHOLD:
        return _multiply_direct(p, q)
    else:
        return _multiply_fft(p, q)

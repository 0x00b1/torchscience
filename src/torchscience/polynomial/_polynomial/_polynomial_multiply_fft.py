"""FFT-based polynomial multiplication for high-degree polynomials.

For high-degree polynomials, FFT-based multiplication has complexity O(n log n)
compared to O(n^2) for direct convolution, making it significantly faster for
large polynomials.
"""

import torch

from ._polynomial import Polynomial, polynomial


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def polynomial_multiply_fft(p: Polynomial, q: Polynomial) -> Polynomial:
    """Multiply two polynomials using FFT-based convolution.

    Uses the Fast Fourier Transform to compute the convolution of polynomial
    coefficients in O(n log n) time, compared to O(n^2) for direct convolution.

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to multiply.

    Returns
    -------
    Polynomial
        Product p * q.

    Notes
    -----
    This function is most efficient for high-degree polynomials (degree > ~64).
    For lower degrees, the direct convolution method may be faster due to
    the overhead of FFT computation.

    The algorithm works by:
    1. Zero-padding both polynomials to length n = deg(p) + deg(q) + 1
    2. Computing FFT of both padded sequences
    3. Multiplying element-wise in frequency domain
    4. Computing inverse FFT to get the result

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> q = polynomial(torch.tensor([4.0, 5.0]))       # 4 + 5x
    >>> r = polynomial_multiply_fft(p, q)              # 4 + 13x + 22x^2 + 15x^3
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
                1,
                dtype=p.dtype,
                device=p.device,
            )
        )

    # Result length
    n_result = n_p + n_q - 1

    # FFT length (round up to power of 2 for efficiency)
    n_fft = _next_power_of_2(n_result)

    # Broadcast batch dimensions
    broadcast_batch = torch.broadcast_shapes(p_batch, q_batch)

    # Convert to plain tensors - the Polynomial subclass has __torch_function__
    # which can interfere with FFT operations
    p_tensor = torch.Tensor(p)
    q_tensor = torch.Tensor(q)

    # Expand to broadcast shape
    p_expanded = p_tensor.expand(*broadcast_batch, n_p)
    q_expanded = q_tensor.expand(*broadcast_batch, n_q)

    # Promote to common dtype for FFT
    # Note: FFT requires float or complex input
    common_dtype = torch.promote_types(p_expanded.dtype, q_expanded.dtype)
    if common_dtype not in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ):
        common_dtype = torch.float64

    p_float = p_expanded.to(common_dtype)
    q_float = q_expanded.to(common_dtype)

    # Compute FFT of both polynomials
    # rfft is more efficient for real inputs
    is_complex = common_dtype in (torch.complex64, torch.complex128)

    if is_complex:
        p_fft = torch.fft.fft(p_float, n=n_fft)
        q_fft = torch.fft.fft(q_float, n=n_fft)
        # Multiply in frequency domain
        result_fft = p_fft * q_fft
        # Inverse FFT
        result_full = torch.fft.ifft(result_fft)
        result = result_full[..., :n_result]
    else:
        p_fft = torch.fft.rfft(p_float, n=n_fft)
        q_fft = torch.fft.rfft(q_float, n=n_fft)
        # Multiply in frequency domain
        result_fft = p_fft * q_fft
        # Inverse FFT
        result_full = torch.fft.irfft(result_fft, n=n_fft)
        result = result_full[..., :n_result]

    # Convert back to original dtype if needed
    original_dtype = torch.promote_types(p.dtype, q.dtype)
    if result.dtype != original_dtype:
        if original_dtype.is_floating_point or original_dtype.is_complex:
            result = result.to(original_dtype)
        else:
            # For integer types, round to nearest integer
            result = result.round().to(original_dtype)

    return polynomial(result)


# Threshold for switching to FFT-based multiplication
# Below this degree, direct convolution is typically faster
FFT_THRESHOLD = 64


def polynomial_multiply_auto(p: Polynomial, q: Polynomial) -> Polynomial:
    """Multiply two polynomials, automatically selecting the best algorithm.

    Uses FFT-based multiplication for high-degree polynomials and direct
    convolution for low-degree polynomials.

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to multiply.

    Returns
    -------
    Polynomial
        Product p * q.

    Notes
    -----
    The threshold for switching to FFT is currently set to degree 64.
    This is a heuristic based on typical performance characteristics, and
    the optimal threshold may vary by hardware.
    """
    n_p = p.shape[-1]
    n_q = q.shape[-1]
    max_degree = max(n_p, n_q) - 1

    if max_degree >= FFT_THRESHOLD:
        return polynomial_multiply_fft(p, q)
    else:
        # Use the standard C++ kernel-based multiplication
        from ._polynomial_multiply import polynomial_multiply

        return polynomial_multiply(p, q)

"""Minimum phase FIR filter conversion."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor


def minimum_phase(
    h: Tensor,
    method: Literal["hilbert", "homomorphic"] = "hilbert",
    n_fft: Optional[int] = None,
) -> Tensor:
    """
    Convert a linear-phase FIR filter to minimum phase.

    Parameters
    ----------
    h : Tensor
        Linear-phase FIR filter coefficients.
    method : {"hilbert", "homomorphic"}, optional
        Conversion method. Default is "hilbert".

        'hilbert'
            This method is designed to be used with equiripple
            filters (e.g., from remez) with unity or zero gain regions.

        'homomorphic'
            This method works best with filters with an odd number of
            taps. The resulting minimum phase filter will have a
            magnitude response that approximates the square root of
            the original filter's magnitude response.

    n_fft : int, optional
        FFT length. Default is 2 ** ceil(log2(2 * (len(h) - 1) / 0.01)).

    Returns
    -------
    h_min : Tensor
        Minimum phase FIR filter coefficients.
        Length is (len(h) + 1) // 2 for odd-length input.

    Notes
    -----
    A minimum phase filter has all zeros inside or on the unit circle,
    resulting in minimum group delay for the given magnitude response.
    """
    h = h.contiguous()
    dtype = h.dtype
    device = h.device
    n = h.shape[0]
    n_half = n // 2

    if n_fft is None:
        # Match scipy's default: 2 ** ceil(log2(2 * (len(h) - 1) / 0.01))
        n_fft = 2 ** int(math.ceil(math.log2(2 * (n - 1) / 0.01)))

    if method == "hilbert":
        return _minimum_phase_hilbert(h, n_fft, dtype, device, n, n_half)
    elif method == "homomorphic":
        return _minimum_phase_homomorphic(h, n_fft, dtype, device, n, n_half)
    else:
        raise ValueError(f"Unknown method: {method}")


def _minimum_phase_hilbert(
    h: Tensor,
    n_fft: int,
    dtype: torch.dtype,
    device: torch.device,
    n: int,
    n_half: int,
) -> Tensor:
    """Minimum phase via Hilbert method (for equiripple filters)."""
    # Compute frequency shift
    w = torch.arange(n_fft, dtype=torch.float64, device=device) * (
        2 * math.pi / n_fft * n_half
    )

    # FFT and apply frequency shift
    H = torch.fft.fft(h.to(torch.float64), n=n_fft)
    H = (H * torch.exp(1j * w)).real

    # Compute passband deviation (dp) and stopband deviation (ds)
    dp = H.max() - 1
    ds = 0 - H.min()

    # Compute scaling factor
    S = 4.0 / (torch.sqrt(1 + dp + ds) + torch.sqrt(1 - dp + ds)) ** 2

    # Shift and scale
    H = H + ds
    H = H * S

    # Take square root (for half-length output)
    H = torch.sqrt(H)

    # Add small value to prevent log explosion
    H = H + 1e-10

    # Apply discrete Hilbert transform for minimum phase
    h_min = _dhtm(H, device)

    # Truncate to output length
    out_len = n_half + n % 2
    h_min = h_min[:out_len]

    return h_min.to(dtype)


def _minimum_phase_homomorphic(
    h: Tensor,
    n_fft: int,
    dtype: torch.dtype,
    device: torch.device,
    n: int,
    n_half: int,
) -> Tensor:
    """Minimum phase via homomorphic method."""
    # Zero-pad and compute magnitude spectrum
    h_temp = torch.fft.fft(h.to(torch.float64), n=n_fft).abs()

    # Avoid log of zero
    min_positive = h_temp[h_temp > 0].min()
    h_temp = h_temp + 1e-7 * min_positive

    # Take 0.5*log(|H|) for half-length output (square root of magnitude)
    h_temp = torch.log(h_temp) * 0.5

    # IDFT
    h_temp = torch.fft.ifft(h_temp).real

    # Multiply by homomorphic filter (double positive frequencies, zero negative)
    win = torch.zeros(n_fft, dtype=torch.float64, device=device)
    win[0] = 1
    stop = n_fft // 2
    win[1:stop] = 2
    if n_fft % 2:
        win[stop] = 1

    h_temp = h_temp * win

    # Transform back
    h_min = torch.fft.ifft(torch.exp(torch.fft.fft(h_temp))).real

    # Truncate to output length
    out_len = n_half + n % 2
    h_min = h_min[:out_len]

    return h_min.to(dtype)


def _dhtm(H: Tensor, device: torch.device) -> Tensor:
    """
    Compute minimum phase impulse response from magnitude spectrum.

    Discrete Hilbert Transform for Minimum phase.
    """
    n_fft = H.shape[0]

    # Take log of magnitude
    log_H = torch.log(H)

    # IDFT to get cepstrum
    cepstrum = torch.fft.ifft(log_H).real

    # Apply minimum phase window (double positive quefrencies, zero negative)
    win = torch.zeros(n_fft, dtype=torch.float64, device=device)
    win[0] = 1
    stop = n_fft // 2
    win[1:stop] = 2
    if n_fft % 2:
        win[stop] = 1

    cepstrum_min = cepstrum * win

    # Transform back
    h_min = torch.fft.ifft(torch.exp(torch.fft.fft(cepstrum_min))).real

    return h_min

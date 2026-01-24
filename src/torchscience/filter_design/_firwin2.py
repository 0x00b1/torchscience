"""Frequency-sampling FIR filter design."""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor


def firwin2(
    num_taps: int,
    frequencies: list[float] | Tensor,
    gains: list[float] | Tensor,
    n_freqs: Optional[int] = None,
    window: str | tuple[str, float] | Callable[[int], Tensor] = "hamming",
    antisymmetric: bool = False,
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Design a FIR filter using frequency sampling.

    From the given frequencies `frequencies` and corresponding gains `gains`,
    this function constructs an FIR filter with an arbitrary frequency response.

    Parameters
    ----------
    num_taps : int
        The number of taps in the FIR filter. This is also the number of
        coefficients returned.
    frequencies : list[float] or Tensor
        The frequency points. Must start at 0, end at 1 (Nyquist), and be
        monotonically increasing. Values are in normalized frequency
        (0 to 1, where 1 is Nyquist).
    gains : list[float] or Tensor
        The gain at each frequency point. Must have same length as frequencies.
    n_freqs : int, optional
        The size of the interpolation grid used for the inverse FFT. If None,
        defaults to 8 * num_taps for smooth interpolation.
    window : str or tuple or callable, optional
        Window function to use. Same options as firwin. Default is "hamming".
    antisymmetric : bool, optional
        If True, design a Type III or IV linear phase filter (antisymmetric).
        If False (default), design a Type I or II filter (symmetric).
    sampling_frequency : float, optional
        The sampling frequency of the system. If specified, frequencies are
        in the same units (e.g., Hz) and will be normalized.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    h : Tensor
        The FIR filter coefficients, shape (num_taps,).

    Notes
    -----
    This function uses linear interpolation to construct a dense frequency
    response from the given points, then uses the inverse FFT to obtain
    the impulse response.

    For antisymmetric filters (differentiators, Hilbert transformers),
    set antisymmetric=True.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter_design import firwin2
    >>> # Design a lowpass filter with smooth transition
    >>> freqs = [0, 0.25, 0.3, 1.0]
    >>> gains = [1, 1, 0, 0]
    >>> h = firwin2(65, freqs, gains)
    >>> h.shape
    torch.Size([65])
    """
    if num_taps < 1:
        raise ValueError(f"num_taps must be at least 1, got {num_taps}")

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Convert to tensors
    if isinstance(frequencies, list):
        freq_tensor = torch.tensor(
            frequencies, dtype=torch.float64, device=device
        )
    else:
        freq_tensor = frequencies.to(dtype=torch.float64, device=device)

    if isinstance(gains, list):
        gain_tensor = torch.tensor(gains, dtype=torch.float64, device=device)
    else:
        gain_tensor = gains.to(dtype=torch.float64, device=device)

    # Validate inputs
    if freq_tensor.numel() != gain_tensor.numel():
        raise ValueError(
            f"frequencies and gains must have same length, "
            f"got {freq_tensor.numel()} and {gain_tensor.numel()}"
        )

    if freq_tensor.numel() < 2:
        raise ValueError("At least 2 frequency points required")

    # Normalize by sampling frequency if provided
    if sampling_frequency is not None:
        nyquist = sampling_frequency / 2.0
        freq_tensor = freq_tensor / nyquist

    # Validate frequency range
    if freq_tensor[0].item() != 0:
        raise ValueError(
            f"First frequency must be 0, got {freq_tensor[0].item()}"
        )
    if freq_tensor[-1].item() != 1:
        raise ValueError(
            f"Last frequency must be 1 (Nyquist), got {freq_tensor[-1].item()}"
        )

    # Check monotonicity
    if torch.any(freq_tensor[1:] <= freq_tensor[:-1]):
        raise ValueError("Frequencies must be strictly increasing")

    # Determine grid size
    if n_freqs is None:
        n_freqs = 8 * num_taps

    # Ensure n_freqs is large enough
    n_freqs = max(n_freqs, num_taps)

    # Make n_freqs even for symmetry in FFT
    if n_freqs % 2 == 1:
        n_freqs += 1

    # Create frequency grid from 0 to 1 (Nyquist)
    freq_grid = torch.linspace(
        0, 1, n_freqs // 2 + 1, dtype=torch.float64, device=device
    )

    # Interpolate gains onto the grid
    gain_grid = _interpolate(freq_tensor, gain_tensor, freq_grid)

    # Build the full frequency response (including negative frequencies)
    if antisymmetric:
        # Type III/IV: antisymmetric impulse response
        # H(-f) = -H(f)*, so for real h, H(-f) = -H(f)
        # The FFT of a real antisymmetric sequence is purely imaginary
        freq_response = 1j * gain_grid
        # Mirror for negative frequencies with sign flip
        # Note: PyTorch doesn't support negative step slicing, use flip instead
        freq_response_full = torch.cat(
            [freq_response, -freq_response[1:-1].flip(0)]
        )
    else:
        # Type I/II: symmetric impulse response
        # H(-f) = H(f)*, so for real h, H(f) is real
        freq_response = gain_grid.to(torch.complex128)
        # Mirror for negative frequencies
        # Note: PyTorch doesn't support negative step slicing, use flip instead
        freq_response_full = torch.cat(
            [freq_response, freq_response[1:-1].flip(0)]
        )

    # Inverse FFT to get impulse response
    h_full = torch.fft.ifft(freq_response_full).real

    # Extract the center num_taps coefficients
    # Shift to center the response
    h_full = torch.fft.fftshift(h_full)

    # Extract centered portion
    center = len(h_full) // 2
    start = center - num_taps // 2
    end = start + num_taps
    h = h_full[start:end]

    # Apply window
    win = _get_window_firwin2(window, num_taps, torch.float64, device)
    h = h * win

    return h.to(dtype)


def _interpolate(x: Tensor, y: Tensor, x_new: Tensor) -> Tensor:
    """Linear interpolation of y at x_new given samples at x."""
    result = torch.zeros_like(x_new)

    for i, xn in enumerate(x_new):
        # Find the interval containing xn
        idx = torch.searchsorted(x, xn)

        if idx == 0:
            result[i] = y[0]
        elif idx >= len(x):
            result[i] = y[-1]
        else:
            # Linear interpolation
            x0, x1 = x[idx - 1], x[idx]
            y0, y1 = y[idx - 1], y[idx]
            t = (xn - x0) / (x1 - x0)
            result[i] = y0 + t * (y1 - y0)

    return result


def _get_window_firwin2(
    window: str | tuple[str, float] | Callable[[int], Tensor],
    num_taps: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Get the window function (same as _get_window in firwin)."""
    if callable(window):
        return window(num_taps).to(dtype=dtype, device=device)

    if isinstance(window, tuple):
        name, param = window
    else:
        name = window
        param = None

    name = name.lower()

    if name in ("rectangular", "boxcar", "rect"):
        return torch.ones(num_taps, dtype=dtype, device=device)

    elif name == "hamming":
        n = torch.arange(num_taps, dtype=dtype, device=device)
        return 0.54 - 0.46 * torch.cos(2 * math.pi * n / (num_taps - 1))

    elif name in ("hann", "hanning"):
        n = torch.arange(num_taps, dtype=dtype, device=device)
        return 0.5 - 0.5 * torch.cos(2 * math.pi * n / (num_taps - 1))

    elif name == "blackman":
        n = torch.arange(num_taps, dtype=dtype, device=device)
        return (
            0.42
            - 0.5 * torch.cos(2 * math.pi * n / (num_taps - 1))
            + 0.08 * torch.cos(4 * math.pi * n / (num_taps - 1))
        )

    elif name in ("bartlett", "triangular"):
        n = torch.arange(num_taps, dtype=dtype, device=device)
        return 1 - torch.abs(2 * n / (num_taps - 1) - 1)

    elif name == "kaiser":
        if param is None:
            param = 8.6
        n = torch.arange(num_taps, dtype=dtype, device=device)
        alpha = (num_taps - 1) / 2.0
        arg = param * torch.sqrt(
            torch.clamp(1 - ((n - alpha) / alpha) ** 2, min=0)
        )
        return torch.special.i0(arg) / torch.special.i0(torch.tensor(param))

    else:
        raise ValueError(f"Unknown window type: {name}")

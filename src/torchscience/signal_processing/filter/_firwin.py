"""Window-based FIR filter design."""

from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from torch import Tensor


def firwin(
    num_taps: int,
    cutoff: float | list[float] | Tensor,
    filter_type: Literal[
        "lowpass", "highpass", "bandpass", "bandstop"
    ] = "lowpass",
    window: str | tuple[str, float] | Callable[[int], Tensor] = "hamming",
    pass_zero: Optional[bool] = None,
    scale: bool = True,
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Design a FIR filter using the window method.

    This function computes the coefficients of a finite impulse response (FIR)
    filter using the window method. The filter has linear phase.

    Parameters
    ----------
    num_taps : int
        Length of the filter (number of coefficients). Must be odd for
        highpass and bandstop filters.
    cutoff : float or list[float] or Tensor
        Cutoff frequency(ies) of the filter. For lowpass and highpass, this
        is a scalar. For bandpass and bandstop, this is a length-2 sequence
        [low, high]. Frequencies are expressed as a fraction of the Nyquist
        frequency (0 to 1), unless sampling_frequency is specified.
    filter_type : {"lowpass", "highpass", "bandpass", "bandstop"}, optional
        Type of filter. Default is "lowpass".
    window : str or tuple or callable, optional
        Window function to use. Can be:
        - A string: "hamming", "hann", "blackman", "bartlett", "rectangular"
        - A tuple (name, param) for parameterized windows like ("kaiser", 8.6)
        - A callable that takes num_taps and returns a window tensor
        Default is "hamming".
    pass_zero : bool, optional
        If True, the filter has gain at frequency 0 (DC). If False, the filter
        has zero gain at DC. If not specified, inferred from filter_type.
    scale : bool, optional
        If True (default), scale the coefficients so the frequency response
        is exactly unity at a certain frequency. For lowpass and bandpass,
        this frequency is 0 (DC). For highpass and bandstop, the gain is
        normalized to 1 at the center of the first passband.
    sampling_frequency : float, optional
        The sampling frequency of the system. If specified, cutoff is in
        the same units (e.g., Hz).
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    h : Tensor
        Coefficients of the FIR filter, shape (num_taps,).

    Notes
    -----
    The window method designs filters by:
    1. Computing the ideal (sinc) impulse response
    2. Truncating to the desired length
    3. Multiplying by a window function to reduce spectral leakage

    The filter has linear phase (constant group delay) equal to
    (num_taps - 1) / 2 samples.

    For Type I FIR filters (odd num_taps, symmetric), the frequency response
    at the Nyquist frequency is not necessarily zero, making them suitable
    for all filter types. For highpass and bandstop filters, num_taps must
    be odd to avoid a zero at Nyquist.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter import firwin
    >>> h = firwin(51, 0.3)  # Lowpass filter with cutoff at 0.3 * Nyquist
    >>> h.shape
    torch.Size([51])
    """
    if num_taps < 1:
        raise ValueError(f"num_taps must be at least 1, got {num_taps}")

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Convert cutoff to tensor
    if isinstance(cutoff, (int, float)):
        cutoff_tensor = torch.tensor(
            [cutoff], dtype=torch.float64, device=device
        )
    elif isinstance(cutoff, list):
        cutoff_tensor = torch.tensor(
            cutoff, dtype=torch.float64, device=device
        )
    else:
        cutoff_tensor = cutoff.to(dtype=torch.float64, device=device)
        # Handle 0-d tensors by converting to 1-d
        if cutoff_tensor.ndim == 0:
            cutoff_tensor = cutoff_tensor.unsqueeze(0)

    # Normalize by sampling frequency if provided
    if sampling_frequency is not None:
        nyquist = sampling_frequency / 2.0
        cutoff_tensor = cutoff_tensor / nyquist

    # Validate cutoff range
    if torch.any(cutoff_tensor <= 0) or torch.any(cutoff_tensor >= 1):
        raise ValueError(
            f"Cutoff frequencies must be in (0, 1), got {cutoff_tensor.tolist()}"
        )

    # Validate cutoff frequencies are strictly increasing
    if len(cutoff_tensor) > 1 and torch.any(
        cutoff_tensor[1:] <= cutoff_tensor[:-1]
    ):
        raise ValueError(
            f"Cutoff frequencies must be strictly increasing, got {cutoff_tensor.tolist()}"
        )

    # Determine pass_zero from filter_type if not specified
    if pass_zero is None:
        pass_zero = filter_type in ("lowpass", "bandstop")

    # Validate num_taps for highpass/bandstop
    if not pass_zero and num_taps % 2 == 0:
        raise ValueError(
            f"For highpass and bandstop filters, num_taps must be odd, got {num_taps}"
        )

    # Build the list of band edges
    # For lowpass: [0, cutoff], passband below cutoff
    # For highpass: [cutoff, 1], passband above cutoff
    # For bandpass: [low, high], passband between low and high
    # For bandstop: [0, low] and [high, 1], passbands outside [low, high]
    bands = _build_bands(cutoff_tensor, pass_zero)

    # Compute the ideal impulse response
    h = _ideal_response(num_taps, bands, device)

    # Apply window
    win = _get_window(window, num_taps, dtype, device)
    h = h * win

    # Scale to get unity gain
    if scale:
        h = _scale_filter(h, bands, pass_zero)

    return h.to(dtype)


def _build_bands(cutoff: Tensor, pass_zero: bool) -> list[tuple[float, float]]:
    """Build list of passband (low, high) pairs."""
    cutoff_list = cutoff.tolist()

    if pass_zero:
        # Starts with a passband at DC
        bands = [(0.0, cutoff_list[0])]
        for i in range(1, len(cutoff_list), 2):
            if i + 1 < len(cutoff_list):
                bands.append((cutoff_list[i], cutoff_list[i + 1]))
            else:
                bands.append((cutoff_list[i], 1.0))
    else:
        # Starts with a stopband at DC
        bands = []
        for i in range(0, len(cutoff_list), 2):
            if i + 1 < len(cutoff_list):
                bands.append((cutoff_list[i], cutoff_list[i + 1]))
            else:
                bands.append((cutoff_list[i], 1.0))

    return bands


def _ideal_response(
    num_taps: int, bands: list[tuple[float, float]], device
) -> Tensor:
    """Compute the ideal (sinc-based) impulse response."""
    # Center of the filter
    alpha = (num_taps - 1) / 2.0

    # Sample indices
    n = torch.arange(num_taps, dtype=torch.float64, device=device)

    # Compute ideal impulse response as sum of sinc functions for each band
    h = torch.zeros(num_taps, dtype=torch.float64, device=device)

    for low, high in bands:
        # Band contribution: (high - low) * sinc((high - low) * (n - alpha))
        # times cos(2 * pi * (high + low) / 2 * (n - alpha))
        # This is equivalent to the difference of two sinc functions
        h += _band_response(n, alpha, low, high)

    return h


def _band_response(n: Tensor, alpha: float, low: float, high: float) -> Tensor:
    """Compute impulse response for a single passband [low, high]."""
    # The ideal lowpass with cutoff f has impulse response:
    # h[n] = f * sinc(f * (n - alpha)) = sin(pi*f*(n-alpha)) / (pi*(n-alpha))
    # For a bandpass [low, high], we use:
    # h[n] = high*sinc(high*(n-alpha)) - low*sinc(low*(n-alpha))

    t = n - alpha

    # Handle the center tap specially to avoid division by zero
    center = t == 0

    # Avoid division by zero
    t_safe = torch.where(center, torch.ones_like(t), t)

    # sinc(x) = sin(pi*x) / (pi*x), so f*sinc(f*t) = sin(pi*f*t) / (pi*t)
    if high > 0:
        arg_high = high * t_safe
        h_high = torch.where(
            center,
            torch.full_like(t, high),
            torch.sin(math.pi * arg_high) / (math.pi * t_safe),
        )
    else:
        h_high = torch.zeros_like(t)

    if low > 0:
        arg_low = low * t_safe
        h_low = torch.where(
            center,
            torch.full_like(t, low),
            torch.sin(math.pi * arg_low) / (math.pi * t_safe),
        )
    else:
        h_low = torch.zeros_like(t)

    return h_high - h_low


def _get_window(
    window: str | tuple[str, float] | Callable[[int], Tensor],
    num_taps: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Get the window function."""
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
            param = 8.6  # Default Kaiser beta
        n = torch.arange(num_taps, dtype=dtype, device=device)
        alpha = (num_taps - 1) / 2.0
        # Kaiser window: I0(beta * sqrt(1 - ((n - alpha) / alpha)^2)) / I0(beta)
        arg = param * torch.sqrt(1 - ((n - alpha) / alpha) ** 2)
        return torch.special.i0(arg) / torch.special.i0(torch.tensor(param))

    else:
        raise ValueError(f"Unknown window type: {name}")


def _scale_filter(
    h: Tensor, bands: list[tuple[float, float]], pass_zero: bool
) -> Tensor:
    """Scale filter coefficients for unity gain."""
    if pass_zero:
        # Scale for unity gain at DC
        scale_freq = 0.0
    else:
        # Scale for unity gain at center of first passband
        # Exception: if the last band includes Nyquist, scale at Nyquist
        last_band = bands[-1]
        if last_band[1] == 1.0:
            # Passband includes Nyquist, scale at Nyquist
            scale_freq = 1.0
        else:
            # Scale at center of first passband
            scale_freq = (bands[0][0] + bands[0][1]) / 2

    # Compute frequency response at scale_freq
    n = torch.arange(len(h), dtype=h.dtype, device=h.device)
    alpha = (len(h) - 1) / 2.0

    # H(f) = sum(h[n] * exp(-j * w * (n - alpha)))
    # For real symmetric filter: H(f) = sum(h[n] * cos(w * (n - alpha)))
    # scale_freq is in normalized Nyquist units (0 to 1 where 1 = Nyquist)
    # w = scale_freq * pi gives angular frequency in rad/sample
    w = math.pi * scale_freq
    gain = torch.sum(h * torch.cos(w * (n - alpha)))

    if abs(gain.item()) > 1e-10:
        h = h / gain

    return h

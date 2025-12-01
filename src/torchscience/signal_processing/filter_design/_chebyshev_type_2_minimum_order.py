"""Chebyshev Type II filter order estimation."""

from __future__ import annotations

import math
from typing import Literal, Optional


def chebyshev_type_2_minimum_order(
    passband_frequency: float | list[float],
    stopband_frequency: float | list[float],
    passband_ripple_db: float,
    stopband_attenuation_db: float,
    filter_type: Literal[
        "lowpass", "highpass", "bandpass", "bandstop"
    ] = "lowpass",
    sampling_frequency: Optional[float] = None,
) -> tuple[int, float]:
    """
    Compute minimum Chebyshev Type II filter order for given specifications.

    Returns the lowest order digital Chebyshev Type II filter that loses no more
    than `passband_ripple_db` dB in the passband and has at least
    `stopband_attenuation_db` dB attenuation in the stopband.

    Parameters
    ----------
    passband_frequency : float or list[float]
        Passband edge frequency(ies). For lowpass and highpass, this is a scalar.
        For bandpass and bandstop, this is a length-2 sequence [low, high].
        Frequencies are expressed as a fraction of the Nyquist frequency (0 to 1),
        unless sampling_frequency is specified.
    stopband_frequency : float or list[float]
        Stopband edge frequency(ies). Same format as passband_frequency.
    passband_ripple_db : float
        Maximum loss in the passband (dB). Must be positive.
    stopband_attenuation_db : float
        Minimum attenuation in the stopband (dB). Must be positive.
    filter_type : {"lowpass", "highpass", "bandpass", "bandstop"}, optional
        Filter type. Default is "lowpass".
    sampling_frequency : float, optional
        The sampling frequency of the digital system. If specified, frequencies
        are in the same units as sampling_frequency (e.g., Hz).

    Returns
    -------
    order : int
        Minimum filter order to meet specifications.
    natural_frequency : float
        Natural frequency for the filter, which equals the stopband edge
        frequency. For bandpass/bandstop, this is a list [low, high].

    Notes
    -----
    Chebyshev Type II filters (inverse Chebyshev) have monotonic passband
    and equiripple stopband. The cutoff frequency is the stopband edge where
    the attenuation reaches the specified level.

    Examples
    --------
    >>> order, wn = chebyshev_type_2_minimum_order(0.2, 0.3, 3, 40)
    >>> order
    4
    """
    # Normalize frequencies if sampling_frequency is provided
    if sampling_frequency is not None:
        nyquist = sampling_frequency / 2.0
        if isinstance(passband_frequency, (list, tuple)):
            passband_frequency = [f / nyquist for f in passband_frequency]
        else:
            passband_frequency = passband_frequency / nyquist
        if isinstance(stopband_frequency, (list, tuple)):
            stopband_frequency = [f / nyquist for f in stopband_frequency]
        else:
            stopband_frequency = stopband_frequency / nyquist

    # Validate frequencies are in (0, 1)
    def validate_freq(f, name):
        if isinstance(f, (list, tuple)):
            if not all(0 < x < 1 for x in f):
                raise ValueError(
                    f"{name} frequencies must be in (0, 1), got {f}"
                )
        else:
            if not 0 < f < 1:
                raise ValueError(
                    f"{name} frequency must be in (0, 1), got {f}"
                )

    validate_freq(passband_frequency, "Passband")
    validate_freq(stopband_frequency, "Stopband")

    # Pre-warp frequencies for bilinear transform
    def prewarp(f):
        if isinstance(f, (list, tuple)):
            return [2 * math.tan(math.pi * x / 2) for x in f]
        return 2 * math.tan(math.pi * f / 2)

    wp_warped = prewarp(passband_frequency)
    ws_warped = prewarp(stopband_frequency)

    # Convert to lowpass prototype frequency ratio
    if filter_type == "lowpass":
        # Stopband edge normalized to 1, passband edge relative
        freq_ratio = ws_warped / wp_warped
    elif filter_type == "highpass":
        # Invert for highpass
        freq_ratio = wp_warped / ws_warped
    elif filter_type == "bandpass":
        wp_low, wp_high = wp_warped
        ws_low, ws_high = ws_warped
        # Center frequency
        w0 = math.sqrt(wp_low * wp_high)
        bw = wp_high - wp_low
        # Transform stopband edges to lowpass prototype
        s_low = abs(ws_low**2 - w0**2) / (ws_low * bw)
        s_high = abs(ws_high**2 - w0**2) / (ws_high * bw)
        freq_ratio = min(s_low, s_high)
    elif filter_type == "bandstop":
        wp_low, wp_high = wp_warped
        ws_low, ws_high = ws_warped
        # Center frequency from stopband
        w0 = math.sqrt(ws_low * ws_high)
        bw = ws_high - ws_low
        # Transform passband edges to lowpass prototype
        p_low = abs(wp_low**2 - w0**2) / (wp_low * bw)
        p_high = abs(wp_high**2 - w0**2) / (wp_high * bw)
        freq_ratio = 1 / min(p_low, p_high)
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}")

    # Compute epsilon values
    eps_p = math.sqrt(10 ** (passband_ripple_db / 10) - 1)
    eps_s = math.sqrt(10 ** (stopband_attenuation_db / 10) - 1)

    # Compute order using Chebyshev formula (same as Type I)
    # order = acosh(eps_s / eps_p) / acosh(freq_ratio)
    order = math.acosh(eps_s / eps_p) / math.acosh(freq_ratio)
    order = int(math.ceil(abs(order)))

    # Natural frequency is the stopband edge for Type II
    wn_digital = stopband_frequency

    # Convert back from normalized if sampling_frequency was provided
    if sampling_frequency is not None:
        if isinstance(wn_digital, list):
            wn_digital = [w * nyquist for w in wn_digital]
        else:
            wn_digital = wn_digital * nyquist

    return order, wn_digital

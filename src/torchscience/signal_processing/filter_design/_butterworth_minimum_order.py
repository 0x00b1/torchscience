"""Butterworth filter order estimation."""

from __future__ import annotations

import math
from typing import Literal, Optional


def butterworth_minimum_order(
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
    Compute minimum Butterworth filter order for given specifications.

    Returns the lowest order digital Butterworth filter that loses no more than
    `passband_ripple_db` dB in the passband and has at least `stopband_attenuation_db`
    dB attenuation in the stopband.

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
        Natural frequency (Butterworth cutoff frequency) for the filter.
        For bandpass/bandstop, this is the geometric mean of the critical frequencies.

    Examples
    --------
    >>> order, wn = butterworth_minimum_order(0.2, 0.3, 3, 40)
    >>> order
    5
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

    # Convert to lowpass prototype frequency
    if filter_type == "lowpass":
        # For lowpass: passband < stopband
        nat_freq_passband = wp_warped
        nat_freq_stopband = ws_warped
    elif filter_type == "highpass":
        # For highpass: passband > stopband, invert frequencies
        nat_freq_passband = 1 / wp_warped
        nat_freq_stopband = 1 / ws_warped
    elif filter_type == "bandpass":
        wp_low, wp_high = wp_warped
        ws_low, ws_high = ws_warped
        # Center frequency (geometric mean)
        w0 = math.sqrt(wp_low * wp_high)
        # Transform to lowpass prototype
        nat_freq_passband = (wp_high - wp_low) / w0
        # For stopband, use the edge that gives the tightest constraint
        s_low = abs(ws_low**2 - w0**2) / (ws_low * (wp_high - wp_low))
        s_high = abs(ws_high**2 - w0**2) / (ws_high * (wp_high - wp_low))
        nat_freq_stopband = min(s_low, s_high)
    elif filter_type == "bandstop":
        wp_low, wp_high = wp_warped
        ws_low, ws_high = ws_warped
        # Center frequency
        w0 = math.sqrt(ws_low * ws_high)
        # Bandwidth in stopband
        bw = ws_high - ws_low
        # Transform to lowpass prototype
        nat_freq_stopband = bw / w0
        # For passband, use the edge that gives the tightest constraint
        p_low = (wp_low * bw) / abs(wp_low**2 - w0**2)
        p_high = (wp_high * bw) / abs(wp_high**2 - w0**2)
        nat_freq_passband = min(p_low, p_high)
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}")

    # Compute order using Butterworth formula
    # order = log((10^(Rs/10) - 1) / (10^(Rp/10) - 1)) / (2 * log(ws/wp))
    num = 10 ** (stopband_attenuation_db / 10) - 1
    den = 10 ** (passband_ripple_db / 10) - 1

    order = math.log10(num / den) / (
        2 * math.log10(nat_freq_stopband / nat_freq_passband)
    )
    order = int(math.ceil(abs(order)))

    # Compute natural frequency (cutoff frequency)
    # Use the passband edge to ensure we meet passband specs
    wn = nat_freq_passband / (
        (10 ** (passband_ripple_db / 10) - 1) ** (1 / (2 * order))
    )

    # Convert back to digital frequency
    if filter_type == "lowpass":
        wn_digital = 2 * math.atan(wn / 2) / math.pi
    elif filter_type == "highpass":
        wn_digital = 2 * math.atan(1 / (wn * 2)) / math.pi
        wn_digital = 1 - wn_digital  # Complement for highpass
    elif filter_type == "bandpass":
        w0 = math.sqrt(wp_warped[0] * wp_warped[1])
        bw = wn * w0
        # Solve for band edges
        w_low = -bw / 2 + math.sqrt((bw / 2) ** 2 + w0**2)
        w_high = bw / 2 + math.sqrt((bw / 2) ** 2 + w0**2)
        wn_digital = [2 * math.atan(w / 2) / math.pi for w in (w_low, w_high)]
    elif filter_type == "bandstop":
        w0 = math.sqrt(ws_warped[0] * ws_warped[1])
        bw = wn * w0
        w_low = -bw / 2 + math.sqrt((bw / 2) ** 2 + w0**2)
        w_high = bw / 2 + math.sqrt((bw / 2) ** 2 + w0**2)
        wn_digital = [2 * math.atan(w / 2) / math.pi for w in (w_low, w_high)]

    # Convert back from normalized if sampling_frequency was provided
    if sampling_frequency is not None:
        if isinstance(wn_digital, list):
            wn_digital = [w * nyquist for w in wn_digital]
        else:
            wn_digital = wn_digital * nyquist

    return order, wn_digital

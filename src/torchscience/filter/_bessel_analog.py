"""Bessel/Thomson analog filter design function."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from ._bessel_prototype import bessel_prototype
from ._lowpass_to_bandpass_zpk import lowpass_to_bandpass_zpk
from ._lowpass_to_bandstop_zpk import lowpass_to_bandstop_zpk
from ._lowpass_to_highpass_zpk import lowpass_to_highpass_zpk
from ._lowpass_to_lowpass_zpk import lowpass_to_lowpass_zpk
from ._zpk_to_ba import zpk_to_ba


def bessel_analog(
    order: int,
    cutoff: Tensor | float | list[float],
    filter_type: Literal[
        "lowpass", "highpass", "bandpass", "bandstop"
    ] = "lowpass",
    normalization: Literal["phase", "delay", "magnitude"] = "phase",
    output: Literal["zpk", "ba"] = "zpk",
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
    """Design an Nth-order analog Bessel/Thomson filter.

    Bessel filters are optimized for maximally flat group delay (linear phase
    in the passband), making them ideal for applications where waveform
    preservation is critical (e.g., audio, pulse shaping).

    Parameters
    ----------
    order : int
        The order of the filter. Must be positive.
    cutoff : Tensor or float or list[float]
        The critical frequency or frequencies in rad/s. For lowpass and highpass,
        this is a scalar. For bandpass and bandstop, this is a length-2 sequence
        [low, high].
    filter_type : {"lowpass", "highpass", "bandpass", "bandstop"}, optional
        The type of filter. Default is "lowpass".
    normalization : {"phase", "delay", "magnitude"}, default "phase"
        Frequency normalization method:
        - "phase": Cutoff is where phase response is -45 degrees * order (default).
        - "delay": Cutoff is where group delay drops to 1/sqrt(2) of DC value.
        - "magnitude": Cutoff is where magnitude response is -3 dB.
    output : {"zpk", "ba"}, optional
        Type of output:
        - "zpk": zeros, poles, gain (default)
        - "ba": numerator/denominator coefficients
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.get_default_dtype().
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    zeros, poles, gain : tuple of Tensors
        Zeros, poles, and gain of the analog filter (if output="zpk").
    numerator, denominator : tuple of Tensors
        Numerator and denominator coefficients of the analog filter (if output="ba").

    Notes
    -----
    Unlike the digital version (bessel_design), this function designs an
    analog filter directly in the s-domain without applying the bilinear
    transform. The cutoff frequency is in rad/s (not normalized).

    Bessel filters have gradual rolloff compared to Butterworth or Chebyshev
    filters, but provide superior time-domain characteristics:
    - Minimal overshoot in step response
    - Preserved waveform shape for signals within the passband
    - Constant group delay in the passband

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import bessel_analog
    >>> z, p, k = bessel_analog(4, 1000.0)  # 4th order lowpass at 1000 rad/s
    >>> z.shape, p.shape
    (torch.Size([0]), torch.Size([4]))

    >>> z, p, k = bessel_analog(4, 1000.0, normalization="magnitude")
    """
    if order < 1:
        raise ValueError(f"Filter order must be positive, got {order}")
    if normalization not in ("phase", "delay", "magnitude"):
        raise ValueError(
            f"normalization must be 'phase', 'delay', or 'magnitude', got '{normalization}'"
        )

    # Convert cutoff to tensor
    if isinstance(cutoff, (int, float)):
        cutoff_tensor = torch.tensor(cutoff, dtype=torch.float64)
    elif isinstance(cutoff, list):
        cutoff_tensor = torch.tensor(cutoff, dtype=torch.float64)
    else:
        cutoff_tensor = cutoff.to(torch.float64)

    # Validate cutoff (must be positive for analog)
    if cutoff_tensor.numel() == 1:
        if cutoff_tensor.item() <= 0:
            raise ValueError(
                f"Cutoff frequency must be positive, got {cutoff_tensor.item()}"
            )
    else:
        if cutoff_tensor.numel() != 2:
            raise ValueError(
                f"For bandpass/bandstop, cutoff must be length 2, got {cutoff_tensor.numel()}"
            )
        if not (0 < cutoff_tensor[0].item() < cutoff_tensor[1].item()):
            raise ValueError(
                f"Cutoff frequencies must satisfy 0 < low < high, got {cutoff_tensor.tolist()}"
            )

    # Validate filter_type
    if filter_type not in ("lowpass", "highpass", "bandpass", "bandstop"):
        raise ValueError(f"Invalid filter_type: {filter_type}")

    # Validate output
    if output not in ("zpk", "ba"):
        raise ValueError(f"Invalid output format: {output}")

    # Get analog Bessel lowpass prototype (cutoff = 1 rad/s)
    z_proto, p_proto, k_proto = bessel_prototype(
        order, normalization=normalization, dtype=dtype, device=device
    )

    # Apply frequency transformation (directly in rad/s, no pre-warping)
    if filter_type == "lowpass":
        z_analog, p_analog, k_analog = lowpass_to_lowpass_zpk(
            z_proto, p_proto, k_proto, cutoff_frequency=cutoff_tensor
        )
    elif filter_type == "highpass":
        z_analog, p_analog, k_analog = lowpass_to_highpass_zpk(
            z_proto, p_proto, k_proto, cutoff_frequency=cutoff_tensor
        )
    elif filter_type == "bandpass":
        bw = cutoff_tensor[1] - cutoff_tensor[0]
        w0 = torch.sqrt(cutoff_tensor[0] * cutoff_tensor[1])
        z_analog, p_analog, k_analog = lowpass_to_bandpass_zpk(
            z_proto, p_proto, k_proto, center_frequency=w0, bandwidth=bw
        )
    elif filter_type == "bandstop":
        bw = cutoff_tensor[1] - cutoff_tensor[0]
        w0 = torch.sqrt(cutoff_tensor[0] * cutoff_tensor[1])
        z_analog, p_analog, k_analog = lowpass_to_bandstop_zpk(
            z_proto, p_proto, k_proto, center_frequency=w0, bandwidth=bw
        )

    # Return in requested format
    if output == "zpk":
        return z_analog, p_analog, k_analog
    elif output == "ba":
        return zpk_to_ba(z_analog, p_analog, k_analog)
    else:
        raise ValueError(f"Invalid output format: {output}")

"""Butterworth analog filter design function."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from ._butterworth_prototype import butterworth_prototype
from ._lowpass_to_bandpass_zpk import lowpass_to_bandpass_zpk
from ._lowpass_to_bandstop_zpk import lowpass_to_bandstop_zpk
from ._lowpass_to_highpass_zpk import lowpass_to_highpass_zpk
from ._lowpass_to_lowpass_zpk import lowpass_to_lowpass_zpk
from ._zpk_to_ba import zpk_to_ba


def butterworth_analog(
    order: int,
    cutoff: Tensor | float | list[float],
    filter_type: Literal[
        "lowpass", "highpass", "bandpass", "bandstop"
    ] = "lowpass",
    output: Literal["zpk", "ba"] = "zpk",
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
    """Design an Nth-order analog Butterworth filter.

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
    The Butterworth filter has maximally flat frequency response in the
    passband. The -3 dB point occurs at the cutoff frequency.

    Unlike the digital version (butterworth_design), this function designs
    an analog filter directly in the s-domain without applying the bilinear
    transform. The cutoff frequency is in rad/s (not normalized).

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_design import butterworth_analog
    >>> z, p, k = butterworth_analog(4, 1000.0)  # 4th order lowpass at 1000 rad/s
    >>> z.shape, p.shape
    (torch.Size([0]), torch.Size([4]))

    >>> z, p, k = butterworth_analog(4, [100.0, 1000.0], filter_type="bandpass")
    >>> p.shape  # Bandpass doubles the order
    torch.Size([8])

    >>> b, a = butterworth_analog(4, 1000.0, output="ba")
    """
    if order < 1:
        raise ValueError(f"Filter order must be positive, got {order}")

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

    # Get analog Butterworth lowpass prototype (cutoff = 1 rad/s)
    z_proto, p_proto, k_proto = butterworth_prototype(
        order, dtype=dtype, device=device
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

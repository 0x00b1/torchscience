"""Chebyshev Type I digital filter design function."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from ._bilinear_transform_zpk import bilinear_transform_zpk
from ._chebyshev_type_1_prototype import chebyshev_type_1_prototype
from ._lowpass_to_bandpass_zpk import lowpass_to_bandpass_zpk
from ._lowpass_to_bandstop_zpk import lowpass_to_bandstop_zpk
from ._lowpass_to_highpass_zpk import lowpass_to_highpass_zpk
from ._lowpass_to_lowpass_zpk import lowpass_to_lowpass_zpk
from ._zpk_to_ba import zpk_to_ba
from ._zpk_to_sos import zpk_to_sos


def chebyshev_type_1_design(
    order: int,
    cutoff: Tensor | float | list[float],
    passband_ripple_db: float,
    filter_type: Literal[
        "lowpass", "highpass", "bandpass", "bandstop"
    ] = "lowpass",
    output: Literal["sos", "zpk", "ba"] = "sos",
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
    """Design an Nth-order digital Chebyshev Type I filter.

    Chebyshev Type I filters have equiripple passband and monotonic stopband.
    They provide steeper rolloff than Butterworth filters at the cost of
    passband ripple.

    Parameters
    ----------
    order : int
        The order of the filter.
    cutoff : Tensor or float or list[float]
        The critical frequency or frequencies. For lowpass and highpass, this
        is a scalar. For bandpass and bandstop, this is a length-2 sequence
        [low, high]. Frequencies are expressed as a fraction of the Nyquist
        frequency (0 to 1), unless sampling_frequency is specified.
    passband_ripple_db : float
        Maximum ripple in the passband in decibels. Must be positive.
        Common values: 0.5 dB, 1 dB, 3 dB.
    filter_type : {"lowpass", "highpass", "bandpass", "bandstop"}, optional
        The type of filter. Default is "lowpass".
    output : {"sos", "zpk", "ba"}, optional
        Type of output:
        - "sos": second-order sections (default, recommended)
        - "zpk": zeros, poles, gain
        - "ba": numerator/denominator
    sampling_frequency : float, optional
        The sampling frequency of the digital system. If specified, cutoff
        is in the same units as sampling_frequency (e.g., Hz).
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.get_default_dtype().
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    sos : Tensor
        Second-order sections representation of the filter (if output="sos").
        Shape: (n_sections, 6) where each row is [b0, b1, b2, a0, a1, a2].
    zeros, poles, gain : tuple of Tensors
        Zeros, poles, and gain of the filter (if output="zpk").
    numerator, denominator : tuple of Tensors
        Numerator and denominator of the filter (if output="ba").

    Notes
    -----
    The Chebyshev Type I filter is designed by:
    1. Creating an analog Chebyshev Type I lowpass prototype
    2. Transforming to the desired frequency band
    3. Converting to digital using the bilinear transform

    The passband ripple parameter controls the tradeoff between passband
    flatness and transition band steepness. Lower ripple means flatter
    passband but wider transition band.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_design import chebyshev_type_1_design
    >>> sos = chebyshev_type_1_design(4, 0.3, passband_ripple_db=1.0)
    >>> sos.shape
    torch.Size([2, 6])
    """
    # Convert cutoff to tensor
    if isinstance(cutoff, (int, float)):
        cutoff_tensor = torch.tensor(cutoff, dtype=torch.float64)
    elif isinstance(cutoff, list):
        cutoff_tensor = torch.tensor(cutoff, dtype=torch.float64)
    else:
        cutoff_tensor = cutoff.to(torch.float64)

    # Normalize by sampling frequency if provided
    if sampling_frequency is not None:
        nyquist = sampling_frequency / 2.0
        cutoff_tensor = cutoff_tensor / nyquist

    # Validate cutoff range (must be 0 < cutoff < 1 for digital)
    if cutoff_tensor.numel() == 1:
        if not (0 < cutoff_tensor.item() < 1):
            raise ValueError(
                f"Cutoff frequency must be between 0 and 1 (Nyquist), got {cutoff_tensor.item()}"
            )
    else:
        if not (0 < cutoff_tensor[0].item() < cutoff_tensor[1].item() < 1):
            raise ValueError(
                f"Cutoff frequencies must satisfy 0 < low < high < 1, got {cutoff_tensor.tolist()}"
            )

    # Get analog Chebyshev Type I lowpass prototype
    z_analog, p_analog, k_analog = chebyshev_type_1_prototype(
        order, passband_ripple_db, dtype=dtype, device=device
    )

    # Pre-warp the cutoff frequencies for bilinear transform
    warped = 4.0 * torch.tan(torch.pi * cutoff_tensor / 2.0)

    # Apply frequency transformation
    if filter_type == "lowpass":
        z_transformed, p_transformed, k_transformed = lowpass_to_lowpass_zpk(
            z_analog, p_analog, k_analog, cutoff_frequency=warped
        )
    elif filter_type == "highpass":
        z_transformed, p_transformed, k_transformed = lowpass_to_highpass_zpk(
            z_analog, p_analog, k_analog, cutoff_frequency=warped
        )
    elif filter_type == "bandpass":
        bw = warped[1] - warped[0]
        w0 = torch.sqrt(warped[0] * warped[1])
        z_transformed, p_transformed, k_transformed = lowpass_to_bandpass_zpk(
            z_analog, p_analog, k_analog, center_frequency=w0, bandwidth=bw
        )
    elif filter_type == "bandstop":
        bw = warped[1] - warped[0]
        w0 = torch.sqrt(warped[0] * warped[1])
        z_transformed, p_transformed, k_transformed = lowpass_to_bandstop_zpk(
            z_analog, p_analog, k_analog, center_frequency=w0, bandwidth=bw
        )
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}")

    # Bilinear transform to digital (sampling_frequency=2 for normalized frequency)
    z_digital, p_digital, k_digital = bilinear_transform_zpk(
        z_transformed, p_transformed, k_transformed, sampling_frequency=2.0
    )

    # Return in requested format
    if output == "zpk":
        return z_digital, p_digital, k_digital
    elif output == "sos":
        return zpk_to_sos(z_digital, p_digital, k_digital)
    elif output == "ba":
        return zpk_to_ba(z_digital, p_digital, k_digital)
    else:
        raise ValueError(f"Invalid output format: {output}")

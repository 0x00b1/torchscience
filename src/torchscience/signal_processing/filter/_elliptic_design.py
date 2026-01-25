"""Elliptic (Cauer) digital filter design function."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from ._bilinear_transform_zpk import bilinear_transform_zpk
from ._elliptic_prototype import elliptic_prototype
from ._lowpass_to_bandpass_zpk import lowpass_to_bandpass_zpk
from ._lowpass_to_bandstop_zpk import lowpass_to_bandstop_zpk
from ._lowpass_to_highpass_zpk import lowpass_to_highpass_zpk
from ._lowpass_to_lowpass_zpk import lowpass_to_lowpass_zpk
from ._zpk_to_ba import zpk_to_ba
from ._zpk_to_sos import zpk_to_sos


def elliptic_design(
    order: int,
    cutoff: Tensor | float | list[float],
    passband_ripple_db: float,
    stopband_attenuation_db: float,
    filter_type: Literal[
        "lowpass", "highpass", "bandpass", "bandstop"
    ] = "lowpass",
    output: Literal["sos", "zpk", "ba"] = "sos",
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
    """Design an Nth-order digital elliptic (Cauer) filter.

    Elliptic filters have equiripple passband and equiripple stopband,
    providing the steepest rolloff for a given filter order. They are
    optimal when strict specifications must be met in both bands.

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
    stopband_attenuation_db : float
        Minimum attenuation in the stopband in decibels. Must be positive.
        Common values: 20 dB, 40 dB, 60 dB.
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
    The elliptic filter is designed by:
    1. Creating an analog elliptic lowpass prototype
    2. Transforming to the desired frequency band
    3. Converting to digital using the bilinear transform

    Elliptic filters provide the steepest rolloff of any classical filter
    type for a given order, but at the cost of ripple in both bands.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter import elliptic_design
    >>> sos = elliptic_design(4, 0.3, passband_ripple_db=1.0, stopband_attenuation_db=40.0)
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

    # Get analog elliptic lowpass prototype
    z_analog, p_analog, k_analog = elliptic_prototype(
        order,
        passband_ripple_db,
        stopband_attenuation_db,
        dtype=dtype,
        device=device,
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

    # Return in requested format, applying dtype/device only if explicitly requested
    if output == "zpk":
        z_out, p_out, k_out = z_digital, p_digital, k_digital
        if dtype is not None or device is not None:
            output_dtype = dtype if dtype is not None else k_out.dtype
            output_device = device if device is not None else k_out.device
            complex_dtype = (
                torch.complex64
                if output_dtype == torch.float32
                else torch.complex128
            )
            z_out = z_out.to(dtype=complex_dtype, device=output_device)
            p_out = p_out.to(dtype=complex_dtype, device=output_device)
            k_out = k_out.to(dtype=output_dtype, device=output_device)
        return z_out, p_out, k_out
    elif output == "sos":
        sos = zpk_to_sos(z_digital, p_digital, k_digital)
        if dtype is not None or device is not None:
            output_dtype = dtype if dtype is not None else sos.dtype
            output_device = device if device is not None else sos.device
            sos = sos.to(dtype=output_dtype, device=output_device)
        return sos
    elif output == "ba":
        b, a = zpk_to_ba(z_digital, p_digital, k_digital)
        if dtype is not None or device is not None:
            output_dtype = dtype if dtype is not None else b.dtype
            output_device = device if device is not None else b.device
            b = b.to(dtype=output_dtype, device=output_device)
            a = a.to(dtype=output_dtype, device=output_device)
        return b, a
    else:
        raise ValueError(f"Invalid output format: {output}")

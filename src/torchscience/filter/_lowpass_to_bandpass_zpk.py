"""Lowpass to bandpass frequency transform for analog filters."""

from typing import Tuple, Union

import torch
from torch import Tensor


def lowpass_to_bandpass_zpk(
    zeros: Tensor,
    poles: Tensor,
    gain: Tensor,
    center_frequency: Union[float, Tensor] = 1.0,
    bandwidth: Union[float, Tensor] = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Transform a lowpass filter to a bandpass filter.

    Performs the analog transformation s -> (s^2 + center_frequency^2) / (bandwidth * s), which
    converts a lowpass filter with cutoff 1 rad/s to a bandpass filter with
    center frequency center_frequency rad/s and bandwidth bandwidth rad/s.

    Parameters
    ----------
    zeros : Tensor
        Zeros of the analog lowpass filter.
    poles : Tensor
        Poles of the analog lowpass filter.
    gain : Tensor
        System gain of the analog lowpass filter.
    center_frequency : float or Tensor
        Center frequency of the bandpass filter (rad/s).
    bandwidth : float or Tensor
        Bandwidth of the bandpass filter (rad/s).

    Returns
    -------
    zeros_new : Tensor
        Zeros of the bandpass filter.
    poles_new : Tensor
        Poles of the bandpass filter.
    gain_new : Tensor
        System gain of the bandpass filter.

    Notes
    -----
    The transformation s -> (s^2 + center_frequency^2) / (bandwidth * s):
    - Doubles the filter order (each pole becomes two poles)
    - Adds (len(poles) - len(zeros)) zeros at s=0
    - Maps the lowpass cutoff to the bandpass edges

    For each pole p_k, the new poles are:
        p_new = (bandwidth * p_k / 2) ± sqrt((bandwidth * p_k / 2)^2 - center_frequency^2)
    """
    if not isinstance(center_frequency, Tensor):
        center_frequency = torch.as_tensor(
            center_frequency, dtype=gain.dtype, device=gain.device
        )
    if not isinstance(bandwidth, Tensor):
        bandwidth = torch.as_tensor(
            bandwidth, dtype=gain.dtype, device=gain.device
        )

    degree_diff = poles.numel() - zeros.numel()
    center_frequency_sq = center_frequency * center_frequency

    # Transform poles: each pole becomes two poles
    # poles_new = (bandwidth * poles / 2) ± sqrt((bandwidth * poles / 2)^2 - center_frequency^2)
    half_bw_p = (bandwidth * poles) / 2
    discriminant = half_bw_p * half_bw_p - center_frequency_sq
    sqrt_disc = torch.sqrt(discriminant.to(poles.dtype))
    poles_new_1 = half_bw_p + sqrt_disc
    poles_new_2 = half_bw_p - sqrt_disc
    poles_new = torch.cat([poles_new_1, poles_new_2])

    # Transform zeros (if any) and add zeros at origin
    if zeros.numel() > 0:
        half_bw_z = (bandwidth * zeros) / 2
        disc_z = half_bw_z * half_bw_z - center_frequency_sq
        sqrt_disc_z = torch.sqrt(disc_z.to(zeros.dtype))
        zeros_new_1 = half_bw_z + sqrt_disc_z
        zeros_new_2 = half_bw_z - sqrt_disc_z
        zeros_transformed = torch.cat([zeros_new_1, zeros_new_2])
    else:
        zeros_transformed = torch.empty(
            0, dtype=poles.dtype, device=poles.device
        )

    # Add zeros at origin
    zeros_at_origin = torch.zeros(
        degree_diff, dtype=poles.dtype, device=poles.device
    )
    zeros_new = torch.cat([zeros_transformed, zeros_at_origin])

    # Adjust gain: gain * bandwidth^(degree_diff)
    gain_new = gain * (bandwidth**degree_diff)

    return zeros_new, poles_new, gain_new

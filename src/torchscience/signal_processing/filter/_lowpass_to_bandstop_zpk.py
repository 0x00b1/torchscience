"""Lowpass to bandstop frequency transform for analog filters."""

from typing import Tuple, Union

import torch
from torch import Tensor


def lowpass_to_bandstop_zpk(
    zeros: Tensor,
    poles: Tensor,
    gain: Tensor,
    center_frequency: Union[float, Tensor] = 1.0,
    bandwidth: Union[float, Tensor] = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Transform a lowpass filter to a bandstop (notch) filter.

    Performs the analog transformation s -> bandwidth * s / (s^2 + center_frequency^2), which
    converts a lowpass filter with cutoff 1 rad/s to a bandstop filter with
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
        Center frequency of the bandstop filter (rad/s).
    bandwidth : float or Tensor
        Bandwidth of the bandstop filter (rad/s).

    Returns
    -------
    zeros_new : Tensor
        Zeros of the bandstop filter.
    poles_new : Tensor
        Poles of the bandstop filter.
    gain_new : Tensor
        System gain of the bandstop filter.

    Notes
    -----
    The transformation s -> bandwidth * s / (s^2 + center_frequency^2):
    - Doubles the filter order (each pole becomes two poles)
    - Adds 2 * (len(poles) - len(zeros)) zeros at ±j*center_frequency
    - Creates a notch at the center frequency

    For each pole p_k, the new poles are:
        p_new = (bandwidth / (2 * p_k)) ± sqrt((bandwidth / (2 * p_k))^2 - center_frequency^2)
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
    # poles_new = (bandwidth / (2 * poles)) ± sqrt((bandwidth / (2 * poles))^2 - center_frequency^2)
    half_bw_over_p = bandwidth / (2 * poles)
    discriminant = half_bw_over_p * half_bw_over_p - center_frequency_sq
    sqrt_disc = torch.sqrt(discriminant.to(poles.dtype))
    poles_new_1 = half_bw_over_p + sqrt_disc
    poles_new_2 = half_bw_over_p - sqrt_disc
    poles_new = torch.cat([poles_new_1, poles_new_2])

    # Transform zeros (if any)
    if zeros.numel() > 0:
        half_bw_over_z = bandwidth / (2 * zeros)
        disc_z = half_bw_over_z * half_bw_over_z - center_frequency_sq
        sqrt_disc_z = torch.sqrt(disc_z.to(zeros.dtype))
        zeros_new_1 = half_bw_over_z + sqrt_disc_z
        zeros_new_2 = half_bw_over_z - sqrt_disc_z
        zeros_transformed = torch.cat([zeros_new_1, zeros_new_2])
    else:
        zeros_transformed = torch.empty(
            0, dtype=poles.dtype, device=poles.device
        )

    # Add zeros at ±j*center_frequency for degree difference
    # Each original pole adds 2 zeros at ±j*center_frequency
    j_center_frequency = 1j * center_frequency
    zeros_at_jwo = (
        torch.stack([j_center_frequency, -j_center_frequency])
        .expand(degree_diff, 2)
        .reshape(-1)
    )
    zeros_at_jwo = zeros_at_jwo.to(poles.dtype)
    zeros_new = torch.cat([zeros_transformed, zeros_at_jwo])

    # Adjust gain
    # For bandstop: gain_new = gain (no change in gain)
    gain_new = gain

    return zeros_new, poles_new, gain_new

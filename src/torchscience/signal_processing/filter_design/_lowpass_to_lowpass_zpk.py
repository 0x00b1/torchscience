"""Lowpass to lowpass frequency transform for analog filters."""

from typing import Tuple, Union

import torch
from torch import Tensor


def lowpass_to_lowpass_zpk(
    zeros: Tensor,
    poles: Tensor,
    gain: Tensor,
    cutoff_frequency: Union[float, Tensor] = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Transform a lowpass filter to a different cutoff frequency.

    Performs the analog frequency scaling s -> s/cutoff_frequency, which scales the
    cutoff frequency from 1 rad/s to cutoff_frequency rad/s.

    Parameters
    ----------
    zeros : Tensor
        Zeros of the analog filter.
    poles : Tensor
        Poles of the analog filter.
    gain : Tensor
        System gain of the analog filter.
    cutoff_frequency : float or Tensor
        New cutoff frequency (rad/s).

    Returns
    -------
    zeros_new : Tensor
        Zeros of the transformed filter.
    poles_new : Tensor
        Poles of the transformed filter.
    gain_new : Tensor
        System gain of the transformed filter.

    Notes
    -----
    The transformation s -> s/cutoff_frequency scales all poles and zeros by cutoff_frequency,
    and adjusts the gain to maintain the correct DC response.

    The gain is multiplied by cutoff_frequency^(len(poles) - len(zeros)) to account for
    the degree difference between numerator and denominator.
    """
    if not isinstance(cutoff_frequency, Tensor):
        cutoff_frequency = torch.as_tensor(
            cutoff_frequency, dtype=gain.dtype, device=gain.device
        )

    # Scale zeros and poles by cutoff_frequency
    zeros_new = zeros * cutoff_frequency
    poles_new = poles * cutoff_frequency

    # Adjust gain: gain * cutoff_frequency^(degree difference)
    degree_diff = poles.numel() - zeros.numel()
    gain_new = gain * (cutoff_frequency**degree_diff)

    return zeros_new, poles_new, gain_new

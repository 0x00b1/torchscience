"""Lowpass to highpass frequency transform for analog filters."""

from typing import Tuple, Union

import torch
from torch import Tensor


def lowpass_to_highpass_zpk(
    zeros: Tensor,
    poles: Tensor,
    gain: Tensor,
    cutoff_frequency: Union[float, Tensor] = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Transform a lowpass filter to a highpass filter.

    Performs the analog transformation s -> cutoff_frequency/s, which converts a
    lowpass filter with cutoff 1 rad/s to a highpass filter with
    cutoff cutoff_frequency rad/s.

    Parameters
    ----------
    zeros : Tensor
        Zeros of the analog lowpass filter.
    poles : Tensor
        Poles of the analog lowpass filter.
    gain : Tensor
        System gain of the analog lowpass filter.
    cutoff_frequency : float or Tensor
        Cutoff frequency of the highpass filter (rad/s).

    Returns
    -------
    zeros_new : Tensor
        Zeros of the highpass filter.
    poles_new : Tensor
        Poles of the highpass filter.
    gain_new : Tensor
        System gain of the highpass filter.

    Notes
    -----
    The transformation s -> cutoff_frequency/s:
    - Maps poles p_k to cutoff_frequency/p_k
    - Maps zeros z_k to cutoff_frequency/z_k
    - Adds (len(poles) - len(zeros)) zeros at s=0
    - Adjusts gain to maintain correct high-frequency response
    """
    if not isinstance(cutoff_frequency, Tensor):
        cutoff_frequency = torch.as_tensor(
            cutoff_frequency, dtype=gain.dtype, device=gain.device
        )

    degree_diff = poles.numel() - zeros.numel()

    # Transform poles: poles_new = cutoff_frequency / poles
    poles_new = cutoff_frequency / poles

    # Transform existing zeros and add zeros at origin
    if zeros.numel() > 0:
        zeros_transformed = cutoff_frequency / zeros
    else:
        zeros_transformed = torch.empty(
            0, dtype=poles.dtype, device=poles.device
        )

    # Add zeros at origin to match degree difference
    zeros_at_origin = torch.zeros(
        degree_diff, dtype=poles.dtype, device=poles.device
    )
    zeros_new = torch.cat([zeros_transformed, zeros_at_origin])

    # Adjust gain using ORIGINAL zeros and poles (not transformed)
    # gain_new = gain * real(prod(-zeros) / prod(-poles))
    # For Butterworth (no zeros): prod(-zeros) = 1
    if zeros.numel() > 0:
        prod_neg_zeros = torch.prod(-zeros)
    else:
        prod_neg_zeros = torch.tensor(
            1.0, dtype=poles.dtype, device=poles.device
        )
    prod_neg_poles = torch.prod(-poles)
    gain_new = gain * torch.real(prod_neg_zeros / prod_neg_poles)

    return zeros_new, poles_new, gain_new

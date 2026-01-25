"""Bilinear transform for analog to digital filter conversion."""

from typing import Tuple, Union

import torch
from torch import Tensor


def bilinear_transform_zpk(
    zeros: Tensor,
    poles: Tensor,
    gain: Tensor,
    sampling_frequency: Union[float, Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Transform an analog filter to a digital filter using bilinear transform.

    The bilinear transform maps the s-plane to the z-plane using:
    s = (2*sampling_frequency) * (z - 1) / (z + 1)

    Parameters
    ----------
    zeros : Tensor
        Zeros of the analog filter.
    poles : Tensor
        Poles of the analog filter.
    gain : Tensor
        System gain of the analog filter.
    sampling_frequency : float or Tensor
        Sampling frequency (Hz).

    Returns
    -------
    zeros_digital : Tensor
        Zeros of the digital filter.
    poles_digital : Tensor
        Poles of the digital filter.
    gain_digital : Tensor
        System gain of the digital filter.

    Notes
    -----
    The bilinear transform:
    - Maps left half-plane (stable analog) to inside unit circle (stable digital)
    - Maps imaginary axis to unit circle
    - Introduces frequency warping: omega_d = 2*sampling_frequency * arctan(omega_a / (2*sampling_frequency))
    - Adds zeros at z=-1 for each degree difference (all-pole analog -> FIR zeros)

    For frequency prewarping (not included here), prewarp the analog
    filter before calling bilinear_transform_zpk.
    """
    if not isinstance(sampling_frequency, Tensor):
        sampling_frequency = torch.as_tensor(
            sampling_frequency, dtype=gain.dtype, device=gain.device
        )

    fs2 = 2 * sampling_frequency  # 2 * sampling frequency

    degree_diff = poles.numel() - zeros.numel()

    # Transform poles: z = (1 + s/(2*fs)) / (1 - s/(2*fs))
    poles_digital = (1 + poles / fs2) / (1 - poles / fs2)

    # Transform existing zeros
    if zeros.numel() > 0:
        zeros_transformed = (1 + zeros / fs2) / (1 - zeros / fs2)
    else:
        zeros_transformed = torch.empty(
            0, dtype=poles.dtype, device=poles.device
        )

    # Add zeros at z=-1 (Nyquist) for degree difference
    zeros_at_nyquist = -torch.ones(
        degree_diff, dtype=poles.dtype, device=poles.device
    )
    zeros_digital = torch.cat([zeros_transformed, zeros_at_nyquist])

    # Adjust gain
    # gain_digital = gain * real(prod(fs2 - zeros) / prod(fs2 - poles))
    if zeros.numel() > 0:
        num = torch.prod(fs2 - zeros)
    else:
        num = torch.tensor(1.0, dtype=poles.dtype, device=poles.device)
    den = torch.prod(fs2 - poles)
    gain_digital = gain * torch.real(num / den)

    return zeros_digital, poles_digital, gain_digital

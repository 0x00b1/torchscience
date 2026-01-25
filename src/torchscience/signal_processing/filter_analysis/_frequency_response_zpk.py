"""Frequency response computation for ZPK filters."""

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def frequency_response_zpk(
    zeros: Tensor,
    poles: Tensor,
    gain: Tensor,
    frequencies: Union[Tensor, int] = 512,
    whole: bool = False,
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute frequency response of a digital filter (ZPK form).

    More numerically stable than frequency_response for filters with
    widely spaced poles/zeros.

    Parameters
    ----------
    zeros : Tensor
        Filter zeros (complex), shape (n_zeros,).
    poles : Tensor
        Filter poles (complex), shape (n_poles,).
    gain : Tensor
        Filter gain (scalar).
    frequencies : Tensor or int, default 512
        If int: Number of frequency points to compute.
        If Tensor: Specific frequency points at which to evaluate.
    whole : bool, default False
        If True and frequencies is int, compute full circle.
    sampling_frequency : float, optional
        If None: frequencies are normalized [0, 1] where 1 = Nyquist.
        If provided: frequencies are in Hz.
    dtype : torch.dtype, optional
        Output dtype for frequency response.
    device : torch.device, optional
        Output device.

    Returns
    -------
    frequencies : Tensor
        Frequency points.
    response : Tensor
        Complex frequency response H(e^{jw}).

    Notes
    -----
    The frequency response is computed as:

    .. math::
        H(e^{j\\omega}) = k \\frac{\\prod_i (e^{j\\omega} - z_i)}{\\prod_i (e^{j\\omega} - p_i)}

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import butterworth_design
    >>> from torchscience.signal_processing.filter_analysis import frequency_response_zpk
    >>> zeros, poles, gain = butterworth_design(4, 0.3, output="zpk")
    >>> freqs, response = frequency_response_zpk(zeros, poles, gain)
    """
    if device is None:
        device = poles.device
    if dtype is None:
        dtype = torch.complex128

    # Generate or validate frequency points
    if isinstance(frequencies, int):
        n_points = frequencies
        if whole:
            max_freq = (
                2.0 if sampling_frequency is None else sampling_frequency
            )
        else:
            max_freq = (
                1.0 if sampling_frequency is None else sampling_frequency / 2.0
            )

        # Use endpoint=False to match scipy behavior
        freq_points = torch.linspace(
            0, max_freq, n_points + 1, dtype=torch.float64, device=device
        )[:-1]
    else:
        freq_points = frequencies.to(dtype=torch.float64, device=device)

    # Convert frequencies to normalized angular frequency
    if sampling_frequency is not None:
        w = 2 * math.pi * freq_points / sampling_frequency
    else:
        w = math.pi * freq_points

    # Compute z = e^{jw}
    z = torch.exp(1j * w)

    # Compute H(z) = k * prod(z - z_i) / prod(z - p_i)
    # Using logarithms for numerical stability with many poles/zeros

    # Numerator: product of (z - z_i) for each frequency point
    if zeros.numel() > 0:
        # Shape: (n_freq, n_zeros)
        z_minus_zeros = z.unsqueeze(-1) - zeros.to(torch.complex128).unsqueeze(
            0
        )
        num = torch.prod(z_minus_zeros, dim=-1)
    else:
        num = torch.ones(len(z), dtype=torch.complex128, device=device)

    # Denominator: product of (z - p_i) for each frequency point
    if poles.numel() > 0:
        # Shape: (n_freq, n_poles)
        z_minus_poles = z.unsqueeze(-1) - poles.to(torch.complex128).unsqueeze(
            0
        )
        den = torch.prod(z_minus_poles, dim=-1)
    else:
        den = torch.ones(len(z), dtype=torch.complex128, device=device)

    # Apply gain
    gain_val = (
        gain.to(torch.complex128)
        if gain.numel() > 0
        else torch.tensor(1.0, dtype=torch.complex128)
    )
    response = gain_val * num / den

    # Determine output dtype for frequencies
    if poles.numel() > 0:
        out_dtype = poles.real.dtype
    elif zeros.numel() > 0:
        out_dtype = zeros.real.dtype
    else:
        out_dtype = torch.float64

    return freq_points.to(out_dtype), response.to(dtype)

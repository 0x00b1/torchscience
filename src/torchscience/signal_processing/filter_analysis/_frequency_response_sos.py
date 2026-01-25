"""Frequency response computation for SOS filters."""

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torchscience.filter._exceptions import SOSNormalizationError


def frequency_response_sos(
    sos: Tensor,
    frequencies: Union[Tensor, int] = 512,
    whole: bool = False,
    sampling_frequency: Optional[float] = None,
    validate: bool = True,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute frequency response of a digital filter (SOS form).

    More numerically stable than frequency_response for high-order filters.

    Parameters
    ----------
    sos : Tensor
        Second-order sections, shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].
    frequencies : Tensor or int, default 512
        If int: Number of frequency points to compute, evenly spaced from
        0 to Nyquist (or 0 to sampling frequency if whole=True).
        If Tensor: Specific frequency points at which to evaluate.
    whole : bool, default False
        If True and frequencies is int, compute from 0 to the full sampling
        frequency instead of 0 to Nyquist. Ignored when frequencies is Tensor.
    sampling_frequency : float, optional
        If None: frequencies are normalized [0, 1] where 1 = Nyquist.
        If provided: frequencies are in Hz.
    validate : bool, default True
        If True, validate that SOS coefficients are properly normalized (a0 = 1).
    dtype : torch.dtype, optional
        Output dtype for frequency response. Defaults to complex64 or complex128.
    device : torch.device, optional
        Output device. Defaults to input device.

    Returns
    -------
    frequencies : Tensor
        Frequency points (normalized or Hz depending on sampling_frequency).
    response : Tensor
        Complex frequency response H(e^{jw}).

    Raises
    ------
    SOSNormalizationError
        If validate=True and a0 != 1 for any section.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import butterworth_design
    >>> from torchscience.signal_processing.filter_analysis import frequency_response_sos
    >>> sos = butterworth_design(4, 0.3)
    >>> freqs, response = frequency_response_sos(sos)
    >>> freqs.shape, response.shape
    (torch.Size([512]), torch.Size([512]))

    >>> # Get magnitude in dB at cutoff
    >>> magnitude_db = 20 * torch.log10(torch.abs(response))
    """
    if device is None:
        device = sos.device
    if dtype is None:
        # Use complex128 for float64, complex64 otherwise
        if sos.dtype == torch.float64:
            dtype = torch.complex128
        else:
            dtype = torch.complex64

    # Validate SOS normalization if requested
    if validate and sos.numel() > 0:
        a0_vals = sos[:, 3]
        if not torch.allclose(a0_vals, torch.ones_like(a0_vals), atol=1e-10):
            raise SOSNormalizationError(
                f"SOS sections must have a0 = 1, got a0 values: {a0_vals.tolist()}"
            )

    # Generate or validate frequency points
    if isinstance(frequencies, int):
        n_points = frequencies
        if whole:
            # 0 to 2*pi (full circle)
            max_freq = (
                2.0 if sampling_frequency is None else sampling_frequency
            )
        else:
            # 0 to pi (half circle, up to Nyquist)
            max_freq = (
                1.0 if sampling_frequency is None else sampling_frequency / 2.0
            )

        # Use endpoint=False to match scipy behavior
        # scipy.signal.sosfreqz uses np.linspace(0, pi, worN, endpoint=False)
        freq_points = torch.linspace(
            0, max_freq, n_points + 1, dtype=torch.float64, device=device
        )[:-1]
    else:
        freq_points = frequencies.to(dtype=torch.float64, device=device)

    # Convert frequencies to normalized angular frequency (0 to pi for Nyquist)
    if sampling_frequency is not None:
        # freq_points are in Hz, convert to normalized [0, pi]
        w = 2 * math.pi * freq_points / sampling_frequency
    else:
        # freq_points are normalized [0, 1] where 1 = Nyquist
        w = math.pi * freq_points

    # Compute z = e^{jw}
    z = torch.exp(1j * w)

    # Compute H(z) = product of all section responses
    # H_k(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
    response = torch.ones(len(z), dtype=dtype, device=device)

    for section in sos:
        b0, b1, b2, a0, a1, a2 = section.to(torch.float64)

        # Numerator: b0 + b1*z^-1 + b2*z^-2
        num = b0 + b1 * z ** (-1) + b2 * z ** (-2)

        # Denominator: a0 + a1*z^-1 + a2*z^-2
        den = a0 + a1 * z ** (-1) + a2 * z ** (-2)

        response = response * (num / den)

    # Return frequencies in original units
    return freq_points.to(sos.dtype), response.to(dtype)

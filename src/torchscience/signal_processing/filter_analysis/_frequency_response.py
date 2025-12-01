"""Frequency response computation for BA (transfer function) filters."""

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def frequency_response(
    numerator: Tensor,
    denominator: Tensor,
    frequencies: Union[Tensor, int] = 512,
    whole: bool = False,
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute frequency response of a digital filter (transfer function form).

    Parameters
    ----------
    numerator : Tensor
        Numerator coefficients (b) in descending powers of z.
    denominator : Tensor
        Denominator coefficients (a) in descending powers of z.
    frequencies : Tensor or int, default 512
        If int: Number of frequency points to compute.
        If Tensor: Specific frequency points at which to evaluate.
    whole : bool, default False
        If True and frequencies is int, compute from 0 to sampling frequency.
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
        H(e^{j\\omega}) = \\frac{\\sum_{k=0}^{N} b_k e^{-j\\omega k}}{\\sum_{k=0}^{M} a_k e^{-j\\omega k}}

    where b_k are numerator coefficients and a_k are denominator coefficients.

    For high-order filters, consider using frequency_response_sos or
    frequency_response_zpk which are more numerically stable.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_analysis import frequency_response
    >>> # Simple first-order lowpass: H(z) = (1 + z^-1) / (1 - 0.5*z^-1)
    >>> b = torch.tensor([1.0, 1.0])
    >>> a = torch.tensor([1.0, -0.5])
    >>> freqs, response = frequency_response(b, a)
    """
    if device is None:
        device = numerator.device
    if dtype is None:
        if numerator.dtype == torch.float64:
            dtype = torch.complex128
        else:
            dtype = torch.complex64

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

    # Compute z^-1 = e^{-jw}
    z_inv = torch.exp(-1j * w)

    # Evaluate polynomials
    # B(z^-1) = b0 + b1*z^-1 + b2*z^-2 + ...
    # For Horner's method with z^-1, we need coefficients in ascending power order
    b = numerator.to(torch.complex128)
    num = _polyval_ascending(b, z_inv)

    # A(z^-1) = a0 + a1*z^-1 + a2*z^-2 + ...
    a = denominator.to(torch.complex128)
    den = _polyval_ascending(a, z_inv)

    response = num / den

    return freq_points.to(numerator.dtype), response.to(dtype)


def _polyval_ascending(coeffs: Tensor, x: Tensor) -> Tensor:
    """Evaluate polynomial at given points for ascending power form.

    For coefficients [c0, c1, c2, ...], evaluates:
    c0 + c1*x + c2*x^2 + ...

    Parameters
    ----------
    coeffs : Tensor
        Polynomial coefficients in ascending power order [c_0, c_1, ..., c_n].
    x : Tensor
        Points at which to evaluate.

    Returns
    -------
    result : Tensor
        Polynomial values at each point.
    """
    if coeffs.numel() == 0:
        return torch.ones_like(x)

    # Use Horner's method with reversed coefficients
    # c0 + c1*x + c2*x^2 = c0 + x*(c1 + x*c2)
    # So we iterate from highest to lowest power
    result = torch.zeros_like(x, dtype=coeffs.dtype)
    for c in reversed(coeffs):
        result = result * x + c

    return result

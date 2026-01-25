"""Frequency response computation for FIR filters."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from ._frequency_response import frequency_response


def frequency_response_fir(
    coefficients: Tensor,
    frequencies: Union[Tensor, int] = 512,
    whole: bool = False,
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute frequency response of an FIR filter.

    Convenience function equivalent to `frequency_response(coefficients, [1.0], ...)`.

    Parameters
    ----------
    coefficients : Tensor
        FIR filter coefficients, shape (num_taps,).
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
    This is a convenience wrapper. FIR filters are all-zero filters with
    denominator [1], so this is equivalent to:

        frequency_response(coefficients, torch.ones(1), ...)

    but avoids the need to construct a trivial denominator tensor.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter import frequency_response_fir
    >>> # Simple moving average (3-tap)
    >>> h = torch.tensor([1/3, 1/3, 1/3])
    >>> freqs, response = frequency_response_fir(h)
    """
    # FIR filters have denominator = [1]
    denominator = torch.ones(
        1, dtype=coefficients.dtype, device=coefficients.device
    )

    return frequency_response(
        coefficients,
        denominator,
        frequencies=frequencies,
        whole=whole,
        sampling_frequency=sampling_frequency,
        dtype=dtype,
        device=device,
    )

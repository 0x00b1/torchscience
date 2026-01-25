"""Impulse and step response computation for digital filters."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def impulse_response(
    numerator: Tensor,
    denominator: Optional[Tensor] = None,
    n_samples: int = 100,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute the impulse response of a digital filter.

    Parameters
    ----------
    numerator : Tensor
        Numerator polynomial coefficients (b) of the transfer function.
    denominator : Tensor, optional
        Denominator polynomial coefficients (a) of the transfer function.
        If not specified, the filter is assumed to be FIR (denominator = [1]).
    n_samples : int, optional
        Number of samples to compute. Default is 100.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    time : Tensor
        Time indices (0, 1, 2, ..., n_samples-1).
    response : Tensor
        Impulse response values at each time index.

    Notes
    -----
    For FIR filters, the impulse response equals the filter coefficients
    (padded with zeros if n_samples > len(numerator)).

    For IIR filters, the impulse response is computed by filtering an
    impulse signal through the filter using the direct form II transposed
    structure.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter import impulse_response
    >>> # FIR filter - impulse response is just the coefficients
    >>> b = torch.tensor([0.25, 0.5, 0.25])
    >>> t, h = impulse_response(b, n_samples=5)
    >>> h
    tensor([0.2500, 0.5000, 0.2500, 0.0000, 0.0000])
    """
    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Convert to float64 for computation
    b = numerator.to(dtype=torch.float64, device=device)

    if denominator is None:
        a = torch.ones(1, dtype=torch.float64, device=device)
    else:
        a = denominator.to(dtype=torch.float64, device=device)

    # Normalize by a[0]
    if a[0].item() != 1.0:
        b = b / a[0]
        a = a / a[0]

    # Create impulse signal
    impulse = torch.zeros(n_samples, dtype=torch.float64, device=device)
    impulse[0] = 1.0

    # Filter the impulse
    response = _lfilter(b, a, impulse)

    # Time indices
    t = torch.arange(n_samples, dtype=dtype, device=device)

    return t, response.to(dtype)


def step_response(
    numerator: Tensor,
    denominator: Optional[Tensor] = None,
    n_samples: int = 100,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute the step response of a digital filter.

    Parameters
    ----------
    numerator : Tensor
        Numerator polynomial coefficients (b) of the transfer function.
    denominator : Tensor, optional
        Denominator polynomial coefficients (a) of the transfer function.
        If not specified, the filter is assumed to be FIR (denominator = [1]).
    n_samples : int, optional
        Number of samples to compute. Default is 100.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    time : Tensor
        Time indices (0, 1, 2, ..., n_samples-1).
    response : Tensor
        Step response values at each time index.

    Notes
    -----
    The step response is the filter's response to a unit step input
    (a signal that is 0 for t < 0 and 1 for t >= 0).

    This is computed by filtering a step signal through the filter,
    which is equivalent to the cumulative sum of the impulse response.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter import step_response
    >>> # FIR filter step response
    >>> b = torch.tensor([0.25, 0.5, 0.25])
    >>> t, s = step_response(b, n_samples=5)
    >>> s
    tensor([0.2500, 0.7500, 1.0000, 1.0000, 1.0000])
    """
    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Convert to float64 for computation
    b = numerator.to(dtype=torch.float64, device=device)

    if denominator is None:
        a = torch.ones(1, dtype=torch.float64, device=device)
    else:
        a = denominator.to(dtype=torch.float64, device=device)

    # Normalize by a[0]
    if a[0].item() != 1.0:
        b = b / a[0]
        a = a / a[0]

    # Create step signal
    step = torch.ones(n_samples, dtype=torch.float64, device=device)

    # Filter the step
    response = _lfilter(b, a, step)

    # Time indices
    t = torch.arange(n_samples, dtype=dtype, device=device)

    return t, response.to(dtype)


def impulse_response_sos(
    sos: Tensor,
    n_samples: int = 100,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute the impulse response of a filter in SOS format.

    Parameters
    ----------
    sos : Tensor
        Second-order sections representation, shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].
    n_samples : int, optional
        Number of samples to compute. Default is 100.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    time : Tensor
        Time indices.
    response : Tensor
        Impulse response values.
    """
    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    sos = sos.to(dtype=torch.float64, device=device)

    # Create impulse signal
    impulse = torch.zeros(n_samples, dtype=torch.float64, device=device)
    impulse[0] = 1.0

    # Filter through each section
    response = impulse
    for i in range(sos.shape[0]):
        b = sos[i, :3]
        a = sos[i, 3:]
        response = _lfilter(b, a, response)

    # Time indices
    t = torch.arange(n_samples, dtype=dtype, device=device)

    return t, response.to(dtype)


def _lfilter(b: Tensor, a: Tensor, x: Tensor) -> Tensor:
    """
    Filter a signal using direct form II transposed structure.

    Parameters
    ----------
    b : Tensor
        Numerator coefficients.
    a : Tensor
        Denominator coefficients (a[0] should be 1).
    x : Tensor
        Input signal.

    Returns
    -------
    y : Tensor
        Filtered signal.
    """
    n_b = len(b)
    n_a = len(a)
    n_x = len(x)

    # Pad coefficients to same length
    n_coef = max(n_b, n_a)
    b_padded = torch.zeros(n_coef, dtype=b.dtype, device=b.device)
    a_padded = torch.zeros(n_coef, dtype=a.dtype, device=a.device)
    b_padded[:n_b] = b
    a_padded[:n_a] = a

    # Initialize output and state
    y = torch.zeros(n_x, dtype=x.dtype, device=x.device)
    z = torch.zeros(n_coef - 1, dtype=x.dtype, device=x.device)

    # Direct form II transposed
    for i in range(n_x):
        # Output
        y[i] = b_padded[0] * x[i] + z[0] if len(z) > 0 else b_padded[0] * x[i]

        # Update state
        for j in range(len(z) - 1):
            z[j] = b_padded[j + 1] * x[i] - a_padded[j + 1] * y[i] + z[j + 1]

        if len(z) > 0:
            z[-1] = b_padded[n_coef - 1] * x[i] - a_padded[n_coef - 1] * y[i]

    return y

"""IIR/FIR filter implementation using direct form II transposed."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor


def lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    dim: int = -1,
    zi: Optional[Tensor] = None,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """
    Filter data along one dimension using an IIR or FIR filter.

    Applies a digital filter to the input signal using the Direct Form II
    Transposed structure. This is equivalent to scipy.signal.lfilter.

    Parameters
    ----------
    b : Tensor
        Numerator coefficients of the filter transfer function.
        Shape (M,) where M is the number of numerator coefficients.
    a : Tensor
        Denominator coefficients of the filter transfer function.
        Shape (N,) where N is the number of denominator coefficients.
        a[0] must be non-zero (it will be used for normalization).
    x : Tensor
        Input signal to filter. Can have any shape.
    dim : int, optional
        The dimension along which to filter. Default is -1 (last dimension).
    zi : Tensor, optional
        Initial conditions for the filter delays. Shape (max(M, N) - 1,).
        If None, zero initial conditions are used.
        If provided, final conditions will also be returned.

    Returns
    -------
    y : Tensor
        Filtered signal, same shape as x.
    zf : Tensor (only if zi is not None)
        Final filter delay values, shape (max(M, N) - 1,).

    Notes
    -----
    The filter is defined by the difference equation:

    .. math::
        a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                    - a[1]*y[n-1] - ... - a[N]*y[n-N]

    The Direct Form II Transposed structure is used for numerical stability.

    This function is fully differentiable with respect to b, a, and x.

    For FIR filters, set a = [1.0] (or just a single element tensor).

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter import lfilter
    >>> # FIR lowpass filter (moving average)
    >>> b = torch.tensor([0.25, 0.5, 0.25])
    >>> a = torch.tensor([1.0])
    >>> x = torch.randn(100)
    >>> y = lfilter(b, a, x)

    >>> # IIR filter with initial conditions
    >>> b = torch.tensor([0.1, 0.2, 0.1])
    >>> a = torch.tensor([1.0, -0.5, 0.1])
    >>> zi = torch.zeros(2)
    >>> y, zf = lfilter(b, a, x, zi=zi)
    """
    # Ensure proper dtypes
    dtype = torch.promote_types(b.dtype, torch.promote_types(a.dtype, x.dtype))
    if not dtype.is_floating_point:
        dtype = torch.float64

    b = b.to(dtype=dtype, device=x.device)
    a = a.to(dtype=dtype, device=x.device)
    x = x.to(dtype=dtype)

    # Normalize by a[0]
    if a[0].item() != 1.0:
        b = b / a[0]
        a = a / a[0]

    # Pad coefficients to same length
    n_b = len(b)
    n_a = len(a)
    n_coef = max(n_b, n_a)

    b_padded = torch.zeros(n_coef, dtype=dtype, device=x.device)
    a_padded = torch.zeros(n_coef, dtype=dtype, device=x.device)
    b_padded[:n_b] = b
    a_padded[:n_a] = a

    # Move filter dimension to last
    x = x.movedim(dim, -1)
    original_shape = x.shape
    n_samples = x.shape[-1]

    # Flatten batch dimensions
    x_flat = x.reshape(-1, n_samples)
    batch_size = x_flat.shape[0]

    # Number of delay elements
    n_delays = n_coef - 1

    # Initialize states
    if zi is None:
        states = torch.zeros(
            batch_size, n_delays, dtype=dtype, device=x.device
        )
        return_states = False
    else:
        zi = zi.to(dtype=dtype, device=x.device)
        states = zi.unsqueeze(0).expand(batch_size, -1).clone()
        return_states = True

    # Output buffer
    y = torch.zeros_like(x_flat)

    # Direct Form II Transposed
    for i in range(n_samples):
        x_i = x_flat[:, i]

        # Output: y[n] = b[0]*x[n] + z[0]
        if n_delays > 0:
            y_i = b_padded[0] * x_i + states[:, 0]
        else:
            y_i = b_padded[0] * x_i

        # Update delay states
        for j in range(n_delays - 1):
            states[:, j] = (
                b_padded[j + 1] * x_i
                - a_padded[j + 1] * y_i
                + states[:, j + 1]
            )

        if n_delays > 0:
            states[:, -1] = (
                b_padded[n_coef - 1] * x_i - a_padded[n_coef - 1] * y_i
            )

        y[:, i] = y_i

    # Reshape back
    y = y.reshape(original_shape)
    y = y.movedim(-1, dim)

    if return_states:
        return y, states[0]  # Return states for first batch element
    return y


def lfiltic(
    b: Tensor,
    a: Tensor,
    y: Tensor,
    x: Optional[Tensor] = None,
) -> Tensor:
    """
    Construct initial conditions for lfilter given past outputs and inputs.

    Given a linear filter with coefficients b and a, and given past outputs y
    and optional past inputs x, compute the initial conditions zi that would
    produce those outputs.

    Parameters
    ----------
    b : Tensor
        Numerator coefficients, shape (M,).
    a : Tensor
        Denominator coefficients, shape (N,). a[0] must be non-zero.
    y : Tensor
        Past outputs, most recent first. Shape (K,) where K >= max(M, N) - 1.
        y[0] is the most recent past output, y[1] is the one before that, etc.
    x : Tensor, optional
        Past inputs, most recent first. Shape (K,) where K >= max(M, N) - 1.
        x[0] is the most recent past input. If None, zeros are assumed.

    Returns
    -------
    zi : Tensor
        Initial conditions suitable for lfilter, shape (max(M, N) - 1,).

    Notes
    -----
    This function is useful for initializing a filter to continue filtering
    a signal from where a previous filtering operation left off.

    The initial conditions are computed to match the Direct Form II Transposed
    structure used by lfilter.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter import lfilter, lfiltic
    >>> b = torch.tensor([0.1, 0.2, 0.1])
    >>> a = torch.tensor([1.0, -0.5, 0.1])
    >>> # Filter in two parts
    >>> x = torch.randn(100)
    >>> y1 = lfilter(b, a, x[:50])
    >>> zi = lfiltic(b, a, y1[-2:].flip(0), x[48:50].flip(0))
    >>> y2 = lfilter(b, a, x[50:], zi=zi)[0]
    """
    dtype = torch.promote_types(b.dtype, torch.promote_types(a.dtype, y.dtype))
    if not dtype.is_floating_point:
        dtype = torch.float64

    device = y.device
    b = b.to(dtype=dtype, device=device)
    a = a.to(dtype=dtype, device=device)
    y = y.to(dtype=dtype)

    # Normalize by a[0]
    if a[0].item() != 1.0:
        b = b / a[0]
        a = a / a[0]

    n_b = len(b)
    n_a = len(a)
    n_coef = max(n_b, n_a)
    n_delays = n_coef - 1

    if n_delays == 0:
        return torch.tensor([], dtype=dtype, device=device)

    # Pad coefficients
    b_padded = torch.zeros(n_coef, dtype=dtype, device=device)
    a_padded = torch.zeros(n_coef, dtype=dtype, device=device)
    b_padded[:n_b] = b
    a_padded[:n_a] = a

    # Handle x
    if x is None:
        x = torch.zeros(n_delays, dtype=dtype, device=device)
    else:
        x = x.to(dtype=dtype, device=device)

    # Pad y and x if necessary
    if len(y) < n_delays:
        y = torch.cat(
            [y, torch.zeros(n_delays - len(y), dtype=dtype, device=device)]
        )
    if len(x) < n_delays:
        x = torch.cat(
            [x, torch.zeros(n_delays - len(x), dtype=dtype, device=device)]
        )

    # Compute initial conditions
    # For Direct Form II Transposed, the state equations are:
    # z[k] = b[k+1]*x - a[k+1]*y + z[k+1]  for k = 0, ..., n_delays-2
    # z[n_delays-1] = b[n_coef-1]*x - a[n_coef-1]*y
    zi = torch.zeros(n_delays, dtype=dtype, device=device)

    # Work backwards from the last state
    for k in range(n_delays - 1, -1, -1):
        # Accumulate contributions from past samples
        zi_k = 0.0
        for j in range(k, n_delays):
            idx = j - k
            if idx < len(y):
                zi_k += b_padded[j + 1] * x[idx] - a_padded[j + 1] * y[idx]
        zi[k] = zi_k

    return zi

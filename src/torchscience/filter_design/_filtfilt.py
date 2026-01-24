"""Zero-phase digital filtering using forward-backward filtering."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from ._lfilter import lfilter
from ._lfilter_zi import lfilter_zi


def filtfilt(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    axis: int = -1,
    padtype: Optional[Literal["odd", "even", "constant"]] = "odd",
    padlen: Optional[int] = None,
) -> Tensor:
    """
    Apply a digital filter forward and backward to a signal.

    This results in zero phase filtering (no phase distortion), at the cost
    of squaring the magnitude response.

    Parameters
    ----------
    b : Tensor
        Numerator polynomial coefficients (feedforward).
    a : Tensor
        Denominator polynomial coefficients (feedback). ``a[0]`` must be
        nonzero. If ``a[0] != 1``, then both ``a`` and ``b`` are normalized
        by ``a[0]``.
    x : Tensor
        Input signal. Can be batched with arbitrary leading dimensions.
    axis : int, optional
        Axis along which to filter. Default is -1 (last axis).
    padtype : {"odd", "even", "constant", None}, optional
        Padding type to reduce transients at the edges. Default is "odd".

        - ``"odd"``: Reflect the signal about the end points and negate it
          (assumes the signal is odd about the end points).
        - ``"even"``: Reflect the signal about the end points
          (assumes the signal is even about the end points).
        - ``"constant"``: Pad with the edge values.
        - ``None``: No padding.
    padlen : int, optional
        Number of elements to pad at each end. Default is
        ``3 * max(len(a), len(b))``. Must be less than
        ``x.shape[axis] - 1``.

    Returns
    -------
    y : Tensor
        Filtered signal with zero phase distortion, same shape as ``x``.

    Notes
    -----
    The filter is applied twice: once forward and once backward. This gives
    zero phase filtering but squares the magnitude response. For a filter
    with frequency response H(w), the overall response is |H(w)|^2.

    The padding is used to reduce edge effects by extending the signal
    before filtering and then truncating the result.

    Fully differentiable with respect to ``b`` and ``x``.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter_design import filtfilt
    >>> # Design a lowpass Butterworth filter
    >>> import scipy.signal
    >>> b_np, a_np = scipy.signal.butter(4, 0.1)
    >>> b = torch.tensor(b_np, dtype=torch.float64)
    >>> a = torch.tensor(a_np, dtype=torch.float64)
    >>> # Filter a noisy signal
    >>> x = torch.randn(100, dtype=torch.float64)
    >>> y = filtfilt(b, a, x)
    """
    # Ensure 1-D coefficient tensors
    b = torch.atleast_1d(b)
    a = torch.atleast_1d(a)

    if b.ndim != 1:
        raise ValueError("b must be a 1-D tensor")
    if a.ndim != 1:
        raise ValueError("a must be a 1-D tensor")

    # Normalize by a[0]
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]

    # Determine filter order
    n_b = b.shape[0]
    n_a = a.shape[0]
    n_order = max(n_b, n_a) - 1

    # Default padlen (scipy uses 3 * max(len(a), len(b)), not 3 * (order))
    if padlen is None:
        padlen = 3 * max(n_b, n_a)

    # Move axis to last position for processing
    x = torch.moveaxis(x, axis, -1)
    original_shape = x.shape
    n_samples = x.shape[-1]

    # Validate padlen
    if padtype is not None and padlen >= n_samples:
        raise ValueError(
            f"padlen ({padlen}) must be less than x.shape[axis] ({n_samples})"
        )

    # Pad signal if needed
    if padtype is not None and padlen > 0:
        x_padded = _pad_signal(x, padtype, padlen)
    else:
        x_padded = x

    # Compute initial conditions
    if n_order > 0:
        zi = lfilter_zi(b, a)

        # Get the first sample for scaling zi (broadcast over batch dims)
        # Shape of zi: (n_order,)
        # Shape of x_padded[..., 0:1]: (..., 1)
        x_first = x_padded[..., 0:1]  # Keep dim for broadcasting

        # Forward pass with initial conditions
        # zi_scaled shape needs to be (..., n_order)
        zi_forward = zi.unsqueeze(0) * x_first  # (..., n_order)
        # Squeeze if 1D input
        if x_padded.ndim == 1:
            zi_forward = zi_forward.squeeze(0)

        y_forward, _ = lfilter(b, a, x_padded, axis=-1, zi=zi_forward)

        # Reverse for backward pass
        y_reversed = y_forward.flip(-1)

        # Backward pass with initial conditions
        y_last = y_reversed[..., 0:1]
        zi_backward = zi.unsqueeze(0) * y_last
        if x_padded.ndim == 1:
            zi_backward = zi_backward.squeeze(0)

        y_backward, _ = lfilter(b, a, y_reversed, axis=-1, zi=zi_backward)

        # Reverse again to get final result
        y = y_backward.flip(-1)
    else:
        # Simple gain filter (order 0) - just apply twice
        y = b[0] * b[0] * x_padded

    # Remove padding
    if padtype is not None and padlen > 0:
        y = y[..., padlen:-padlen]

    # Restore original axis position
    y = torch.moveaxis(y, -1, axis)

    return y


def _pad_signal(
    x: Tensor,
    padtype: Literal["odd", "even", "constant"],
    padlen: int,
) -> Tensor:
    """
    Pad signal for filtfilt edge handling.

    Parameters
    ----------
    x : Tensor
        Input signal with samples along last axis.
    padtype : {"odd", "even", "constant"}
        Type of padding.
    padlen : int
        Number of samples to pad at each end.

    Returns
    -------
    x_padded : Tensor
        Padded signal.
    """
    if padtype == "odd":
        # Odd extension: reflect and negate about endpoints
        # Left: 2*x[0] - x[padlen:0:-1] (reversed and negated offset from x[0])
        # Right: 2*x[-1] - x[-2:-padlen-2:-1] (reversed and negated offset from x[-1])
        # PyTorch doesn't support negative step, so use flip
        left = 2 * x[..., 0:1] - x[..., 1 : padlen + 1].flip(-1)
        right = 2 * x[..., -1:] - x[..., -padlen - 1 : -1].flip(-1)
        return torch.cat([left, x, right], dim=-1)

    elif padtype == "even":
        # Even extension: reflect about endpoints
        # Left: x[padlen:0:-1]
        # Right: x[-2:-padlen-2:-1]
        left = x[..., 1 : padlen + 1].flip(-1)
        right = x[..., -padlen - 1 : -1].flip(-1)
        return torch.cat([left, x, right], dim=-1)

    elif padtype == "constant":
        # Constant extension: pad with edge values
        left = x[..., 0:1].expand(*x.shape[:-1], padlen)
        right = x[..., -1:].expand(*x.shape[:-1], padlen)
        return torch.cat([left, x, right], dim=-1)

    else:
        raise ValueError(f"Unknown padtype: {padtype}")

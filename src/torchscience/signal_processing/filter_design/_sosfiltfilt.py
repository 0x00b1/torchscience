"""Zero-phase digital filtering using second-order sections."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from ._sosfilt import sosfilt
from ._sosfilt_zi import sosfilt_zi


def sosfiltfilt(
    sos: Tensor,
    x: Tensor,
    axis: int = -1,
    padtype: Optional[Literal["odd", "even", "constant"]] = "odd",
    padlen: Optional[int] = None,
) -> Tensor:
    """
    Apply a digital filter forward and backward using second-order sections.

    This results in zero phase filtering (no phase distortion), at the cost
    of squaring the magnitude response. Using second-order sections (SOS)
    provides better numerical stability than direct transfer function (b, a)
    for high-order filters.

    Parameters
    ----------
    sos : Tensor
        Second-order sections, shape ``(n_sections, 6)``. Each row contains
        ``[b0, b1, b2, a0, a1, a2]`` for one biquad section. Typically
        ``a0 = 1.0`` for each section.
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
        Number of elements to pad at each end. Default matches scipy:
        ``3 * (2 * n_sections + 1 - n_zeros_at_origin)`` where
        ``n_zeros_at_origin`` is the minimum of zeros at origin in
        numerator (b2==0) and denominator (a2==0) coefficients.
        Must be less than ``x.shape[axis] - 1``.

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

    The SOS format is preferred over transfer function (b, a) for high-order
    filters because it is more numerically stable. Each second-order section
    is:

    .. math::
        H_k(z) = \\frac{b_{k,0} + b_{k,1} z^{-1} + b_{k,2} z^{-2}}
                      {a_{k,0} + a_{k,1} z^{-1} + a_{k,2} z^{-2}}

    Fully differentiable with respect to ``sos`` and ``x``.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_design import sosfiltfilt
    >>> # Design a lowpass Butterworth filter
    >>> import scipy.signal
    >>> sos_np = scipy.signal.butter(4, 0.1, output='sos')
    >>> sos = torch.tensor(sos_np, dtype=torch.float64)
    >>> # Filter a noisy signal
    >>> x = torch.randn(100, dtype=torch.float64)
    >>> y = sosfiltfilt(sos, x)
    """
    # Validate SOS shape
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError("sos must be shape (n_sections, 6)")

    n_sections = sos.shape[0]

    # Filter order is 2 * n_sections (each section is second-order)
    n_order = 2 * n_sections

    # Default padlen matches scipy:
    # ntaps = 2 * n_sections + 1
    # ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    # padlen = 3 * ntaps
    if padlen is None:
        ntaps = 2 * n_sections + 1
        # Count zeros at origin (b2 == 0 or a2 == 0)
        n_zeros_num = (sos[:, 2] == 0).sum().item()
        n_zeros_den = (sos[:, 5] == 0).sum().item()
        ntaps -= min(n_zeros_num, n_zeros_den)
        padlen = 3 * ntaps

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
        zi = sosfilt_zi(sos)

        # Get the first sample for scaling zi (broadcast over batch dims)
        # Shape of zi: (n_sections, 2)
        # Shape of x_padded[..., 0:1]: (..., 1)
        x_first = x_padded[..., 0:1]  # Keep dim for broadcasting

        # Forward pass with initial conditions
        # zi_scaled shape needs to be (..., n_sections, 2)
        zi_forward = zi.unsqueeze(0) * x_first.unsqueeze(
            -1
        )  # (..., n_sections, 2)
        # Squeeze if 1D input
        if x_padded.ndim == 1:
            zi_forward = zi_forward.squeeze(0)

        y_forward, _ = sosfilt(sos, x_padded, axis=-1, zi=zi_forward)

        # Reverse for backward pass
        y_reversed = y_forward.flip(-1)

        # Backward pass with initial conditions
        y_last = y_reversed[..., 0:1]
        zi_backward = zi.unsqueeze(0) * y_last.unsqueeze(-1)
        if x_padded.ndim == 1:
            zi_backward = zi_backward.squeeze(0)

        y_backward, _ = sosfilt(sos, y_reversed, axis=-1, zi=zi_backward)

        # Reverse again to get final result
        y = y_backward.flip(-1)
    else:
        # This shouldn't happen with valid SOS
        y = x_padded

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
    Pad signal for sosfiltfilt edge handling.

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
        left = 2 * x[..., 0:1] - x[..., 1 : padlen + 1].flip(-1)
        right = 2 * x[..., -1:] - x[..., -padlen - 1 : -1].flip(-1)
        return torch.cat([left, x, right], dim=-1)

    elif padtype == "even":
        # Even extension: reflect about endpoints
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

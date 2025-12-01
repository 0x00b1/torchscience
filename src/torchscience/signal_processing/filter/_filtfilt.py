"""Zero-phase digital filtering."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from ._lfilter import lfilter


def filtfilt(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    dim: int = -1,
    padtype: Literal["odd", "even", "constant", None] = "odd",
    padlen: Optional[int] = None,
) -> Tensor:
    """
    Apply a digital filter forward and backward to a signal (zero-phase).

    This function applies the filter twice: once forward and once backward.
    The result has zero phase distortion, which is particularly useful for
    preserving the timing of features in the signal.

    Parameters
    ----------
    b : Tensor
        Numerator coefficients of the filter.
    a : Tensor
        Denominator coefficients of the filter.
    x : Tensor
        Input signal to be filtered.
    dim : int, optional
        The dimension along which to filter. Default is -1.
    padtype : {"odd", "even", "constant", None}, optional
        Type of padding to use for the signal ends:
        - "odd": Reflect signal about its endpoints with sign flip (default)
        - "even": Reflect signal about its endpoints
        - "constant": Pad with edge values
        - None: No padding (may cause transient effects at boundaries)
    padlen : int, optional
        Number of samples to pad at each end. If None, uses 3 * max(len(a), len(b)).
        Must be less than the signal length.

    Returns
    -------
    y : Tensor
        Filtered signal with zero phase distortion, same shape as x.

    Notes
    -----
    The effective filter order is doubled by the forward-backward application.
    This provides steeper roll-off but also amplifies the filter's effect
    (e.g., a 3dB cutoff becomes approximately 6dB).

    The padding helps minimize transient effects at the signal boundaries.
    The "odd" padding is generally recommended as it minimizes edge effects.

    This function is differentiable with respect to x but not with respect to
    the filter coefficients b and a (due to the flip operation).

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter import filtfilt
    >>> from torchscience.signal_processing.filter_design import butterworth_design, zpk_to_ba
    >>> # Design a lowpass filter
    >>> z, p, k = butterworth_design(4, 0.1, output='zpk')
    >>> b, a = zpk_to_ba(z, p, k)
    >>> # Apply zero-phase filtering
    >>> x = torch.randn(1000)
    >>> y = filtfilt(b, a, x)
    """
    # Get proper dtype
    dtype = torch.promote_types(b.dtype, torch.promote_types(a.dtype, x.dtype))
    if not dtype.is_floating_point:
        dtype = torch.float64

    b = b.to(dtype=dtype, device=x.device)
    a = a.to(dtype=dtype, device=x.device)
    x = x.to(dtype=dtype)

    # Move filter dimension to last
    x = x.movedim(dim, -1)
    original_shape = x.shape
    n_samples = x.shape[-1]

    # Determine padding length
    n_coef = max(len(b), len(a))
    if padlen is None:
        padlen = 3 * n_coef

    if padlen > 0 and padlen >= n_samples:
        raise ValueError(
            f"padlen ({padlen}) must be less than signal length ({n_samples})"
        )

    # Flatten batch dimensions
    x_flat = x.reshape(-1, n_samples)
    batch_size = x_flat.shape[0]

    # Pad the signal
    if padtype is not None and padlen > 0:
        x_padded = _pad_signal(x_flat, padlen, padtype)
    else:
        x_padded = x_flat

    # Compute initial conditions for forward pass
    zi_forward = _compute_zi(b, a).to(dtype=dtype, device=x.device)

    # Apply forward filter with initial conditions based on first value
    # Scale zi by the first sample value to reduce transients
    y_forward_list = []
    zf_list = []
    for i in range(batch_size):
        xi = x_padded[i : i + 1, :]
        zi_scaled = zi_forward * xi[0, 0]
        y_fwd, zf = lfilter(b, a, xi.squeeze(0), dim=-1, zi=zi_scaled)
        y_forward_list.append(y_fwd)
        zf_list.append(zf)

    y_forward = torch.stack(y_forward_list, dim=0)

    # Reverse the signal
    y_reversed = y_forward.flip(-1)

    # Apply backward filter with initial conditions based on last value
    y_backward_list = []
    for i in range(batch_size):
        yi = y_reversed[i : i + 1, :]
        zi_scaled = zi_forward * yi[0, 0]
        y_bwd, _ = lfilter(b, a, yi.squeeze(0), dim=-1, zi=zi_scaled)
        y_backward_list.append(y_bwd)

    y_backward = torch.stack(y_backward_list, dim=0)

    # Reverse back to original direction
    y = y_backward.flip(-1)

    # Remove padding
    if padtype is not None and padlen > 0:
        y = y[:, padlen:-padlen]

    # Reshape back
    y = y.reshape(original_shape)
    y = y.movedim(-1, dim)

    return y


def _pad_signal(
    x: Tensor,
    padlen: int,
    padtype: str,
) -> Tensor:
    """Pad signal for filtfilt to reduce edge effects."""
    n_samples = x.shape[-1]

    # Note: PyTorch doesn't support negative step slicing, so we use flip()
    if padtype == "odd":
        # Reflect about endpoints with sign flip
        # Creates continuity in first derivative
        left_pad = 2 * x[:, 0:1] - x[:, 1 : padlen + 1].flip(-1)
        right_pad = 2 * x[:, -1:] - x[:, -padlen - 1 : -1].flip(-1)
    elif padtype == "even":
        # Reflect about endpoints
        left_pad = x[:, 1 : padlen + 1].flip(-1)
        right_pad = x[:, -padlen - 1 : -1].flip(-1)
    elif padtype == "constant":
        # Pad with edge values
        left_pad = x[:, 0:1].expand(-1, padlen)
        right_pad = x[:, -1:].expand(-1, padlen)
    else:
        raise ValueError(f"Unknown padtype: {padtype}")

    return torch.cat([left_pad, x, right_pad], dim=-1)


def _compute_zi(b: Tensor, a: Tensor) -> Tensor:
    """
    Compute the initial conditions for lfilter that minimize transients.

    This implements scipy's lfilter_zi: computes the steady-state filter
    delays for a step input. When these initial conditions are scaled by
    the first sample value, they minimize edge transients.

    Parameters
    ----------
    b : Tensor
        Numerator coefficients.
    a : Tensor
        Denominator coefficients.

    Returns
    -------
    zi : Tensor
        Initial conditions, shape (max(len(b), len(a)) - 1,).
    """
    dtype = torch.promote_types(b.dtype, a.dtype)
    device = b.device

    # Normalize
    if a[0].item() != 1.0:
        b = b / a[0]
        a = a / a[0]

    n_b = len(b)
    n_a = len(a)
    n = max(n_b, n_a)

    if n <= 1:
        return torch.tensor([], dtype=dtype, device=device)

    # Pad coefficients to same length
    b_padded = torch.zeros(n, dtype=dtype, device=device)
    a_padded = torch.zeros(n, dtype=dtype, device=device)
    b_padded[:n_b] = b
    a_padded[:n_a] = a

    # Build the companion matrix A = companion(a).T
    # The companion matrix of polynomial [1, a1, a2, ...] has:
    # - First row: [-a1, -a2, ...]
    # - Subdiagonal: all 1s
    # When transposed:
    # - First column: [-a1, -a2, ...]
    # - Superdiagonal: all 1s
    A = torch.zeros(n - 1, n - 1, dtype=dtype, device=device)
    A[:, 0] = -a_padded[1:]  # First column is -a[1:]
    for i in range(n - 2):
        A[i, i + 1] = 1.0  # Superdiagonal

    # Build I - A
    IminusA = torch.eye(n - 1, dtype=dtype, device=device) - A

    # B = b[1:] - a[1:] * b[0]
    B = b_padded[1:] - a_padded[1:] * b_padded[0]

    # Solve (I - A) @ zi = B
    try:
        zi = torch.linalg.solve(IminusA, B)
    except RuntimeError:
        # Fallback: use pseudo-inverse for singular systems
        zi = torch.linalg.lstsq(IminusA, B).solution

    return zi

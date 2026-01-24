"""Zero-phase digital filtering using second-order sections."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from ._sosfilt import sosfilt


def sosfiltfilt(
    sos: Tensor,
    x: Tensor,
    dim: int = -1,
    padtype: Literal["odd", "even", "constant", None] = "odd",
    padlen: Optional[int] = None,
) -> Tensor:
    """
    Apply a digital filter forward and backward using second-order sections.

    This function applies the filter twice: once forward and once backward.
    The result has zero phase distortion. This is the SOS equivalent of filtfilt.

    Parameters
    ----------
    sos : Tensor
        Second-order sections representation, shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].
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
        Number of samples to pad at each end. If None, uses 3 * (2 * n_sections),
        which accounts for 2 delay elements per section.

    Returns
    -------
    y : Tensor
        Filtered signal with zero phase distortion, same shape as x.

    Notes
    -----
    Using second-order sections (SOS) format is generally preferred over
    transfer function (BA) format for high-order filters, as it provides
    better numerical stability.

    The effective filter order is doubled by the forward-backward application.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import sosfiltfilt
    >>> from torchscience.filter_design import butterworth_design
    >>> # Design a lowpass filter
    >>> sos = butterworth_design(4, 0.1)
    >>> # Apply zero-phase filtering
    >>> x = torch.randn(1000)
    >>> y = sosfiltfilt(sos, x)
    """
    dtype = x.dtype
    if not dtype.is_floating_point:
        dtype = torch.float64

    sos = sos.to(dtype=dtype, device=x.device)
    x = x.to(dtype=dtype)

    # Move filter dimension to last
    x = x.movedim(dim, -1)
    original_shape = x.shape
    n_samples = x.shape[-1]

    n_sections = sos.shape[0]

    # Determine padding length
    # scipy uses ntaps * 3 where ntaps = 2 * n_sections + 1
    # adjusted for any zero coefficients
    if padlen is None:
        ntaps = 2 * n_sections + 1
        # Adjust for zero b2 and a2 coefficients
        zeros_b2 = (sos[:, 2] == 0).sum().item()
        zeros_a2 = (sos[:, 5] == 0).sum().item()
        ntaps -= min(zeros_b2, zeros_a2)
        padlen = ntaps * 3

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

    # Compute initial conditions for each section
    zi_all = _compute_sos_zi(sos).to(dtype=dtype, device=x.device)

    # Apply forward filter with initial conditions
    y_forward_list = []
    for i in range(batch_size):
        xi = x_padded[i : i + 1, :]
        # Scale zi by the first sample value
        zi_scaled = zi_all * xi[0, 0]
        y_fwd, _ = sosfilt(sos, xi.squeeze(0), dim=-1, zi=zi_scaled)
        y_forward_list.append(y_fwd)

    y_forward = torch.stack(y_forward_list, dim=0)

    # Reverse the signal
    y_reversed = y_forward.flip(-1)

    # Apply backward filter with initial conditions
    y_backward_list = []
    for i in range(batch_size):
        yi = y_reversed[i : i + 1, :]
        # Scale zi by the first sample value of reversed signal
        zi_scaled = zi_all * yi[0, 0]
        y_bwd, _ = sosfilt(sos, yi.squeeze(0), dim=-1, zi=zi_scaled)
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
    """Pad signal for sosfiltfilt to reduce edge effects."""
    # Note: PyTorch doesn't support negative step slicing, so we use flip()
    if padtype == "odd":
        # Reflect about endpoints with sign flip
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


def _compute_sos_zi(sos: Tensor) -> Tensor:
    """
    Compute initial conditions for each SOS section that minimize transients.

    This implements scipy's sosfilt_zi: for each section, compute the
    steady-state filter delays using lfilter_zi, and scale by the
    cumulative DC gain of previous sections.

    Parameters
    ----------
    sos : Tensor
        Second-order sections, shape (n_sections, 6).

    Returns
    -------
    zi : Tensor
        Initial conditions, shape (n_sections, 2).
    """
    n_sections = sos.shape[0]
    dtype = sos.dtype
    device = sos.device

    zi = torch.zeros(n_sections, 2, dtype=dtype, device=device)
    scale = 1.0

    for i in range(n_sections):
        b = sos[i, :3]
        a = sos[i, 3:]

        # Compute lfilter_zi for this section
        zi_section = _lfilter_zi_2nd_order(b, a)
        zi[i, :] = scale * zi_section

        # Update scale by DC gain of this section: H(1) = sum(b) / sum(a)
        scale = scale * b.sum() / a.sum()

    return zi


def _lfilter_zi_2nd_order(b: Tensor, a: Tensor) -> Tensor:
    """
    Compute lfilter_zi for a second-order section.

    This is a specialized version for 2nd order filters that avoids
    the general linear algebra solve.
    """
    dtype = b.dtype
    device = b.device

    # Normalize by a[0]
    if a[0].item() != 1.0:
        b = b / a[0]
        a = a / a[0]

    # For a 2nd order section, the companion matrix A = companion(a).T is:
    # A = [[-a1, 1],
    #      [-a2, 0]]
    # I - A = [[1 + a1, -1],
    #          [a2,     1]]
    # B = [b1 - a1*b0, b2 - a2*b0]

    # Solve (I - A) @ zi = B
    # [1 + a1, -1] [z0]   [b1 - a1*b0]
    # [a2,     1 ] [z1] = [b2 - a2*b0]

    # From second equation: a2*z0 + z1 = b2 - a2*b0
    # => z1 = b2 - a2*b0 - a2*z0

    # From first equation: (1 + a1)*z0 - z1 = b1 - a1*b0
    # Substitute z1: (1 + a1)*z0 - (b2 - a2*b0 - a2*z0) = b1 - a1*b0
    # (1 + a1 + a2)*z0 = b1 - a1*b0 + b2 - a2*b0
    # z0 = (b1 + b2 - (a1 + a2)*b0) / (1 + a1 + a2)

    denom = 1.0 + a[1] + a[2]
    z0 = (b[1] + b[2] - (a[1] + a[2]) * b[0]) / denom
    z1 = b[2] - a[2] * b[0] - a[2] * z0

    return torch.stack([z0, z1])

"""Batched filter design variants for parallel filter design.

This module provides vectorized filter design functions that efficiently
design multiple filters in parallel with different cutoff frequencies.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from ._butterworth_design import butterworth_design
from ._chebyshev_type_1_design import chebyshev_type_1_design
from ._chebyshev_type_2_design import chebyshev_type_2_design
from ._firwin import firwin
from ._lfilter import lfilter
from ._sosfilt import sosfilt


def batched_butterworth_design(
    order: int,
    cutoffs: Tensor,
    filter_type: Literal[
        "lowpass", "highpass", "bandpass", "bandstop"
    ] = "lowpass",
    output: Literal["sos"] = "sos",
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Design multiple Butterworth filters with different cutoff frequencies.

    Efficiently designs multiple filters in parallel by iterating over the
    batch dimension. Supports gradient computation for optimization.

    Parameters
    ----------
    order : int
        Filter order (same for all filters).
    cutoffs : Tensor
        Cutoff frequencies. Shape ``(batch,)`` for lowpass/highpass filters,
        or ``(batch, 2)`` for bandpass/bandstop filters. Frequencies are
        expressed as a fraction of the Nyquist frequency (0 to 1), unless
        ``sampling_frequency`` is specified.
    filter_type : {"lowpass", "highpass", "bandpass", "bandstop"}, optional
        Filter type. Default is "lowpass".
    output : {"sos"}, optional
        Output format. Currently only "sos" is supported for batched design.
    sampling_frequency : float, optional
        The sampling frequency of the digital system. If specified, cutoffs
        are in the same units (e.g., Hz).
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    filters : Tensor
        Batched filter coefficients. Shape ``(batch, n_sections, 6)`` for SOS
        output, where each section contains ``[b0, b1, b2, a0, a1, a2]``.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import batched_butterworth_design
    >>> cutoffs = torch.linspace(0.1, 0.4, 10)
    >>> filters = batched_butterworth_design(4, cutoffs)
    >>> filters.shape
    torch.Size([10, 2, 6])

    Design bandpass filters with varying frequency bands:

    >>> cutoffs = torch.tensor([[0.1, 0.3], [0.2, 0.4], [0.15, 0.35]])
    >>> filters = batched_butterworth_design(2, cutoffs, filter_type="bandpass")
    >>> filters.shape
    torch.Size([3, 2, 6])
    """
    if output != "sos":
        raise ValueError(
            f"Batched design only supports 'sos' output, got '{output}'"
        )

    if cutoffs.numel() == 0:
        raise ValueError("cutoffs tensor cannot be empty")

    # Ensure cutoffs is at least 1D
    cutoffs = torch.atleast_1d(cutoffs)

    # Determine if bandpass/bandstop (requires 2D cutoffs)
    is_band_filter = filter_type in ("bandpass", "bandstop")

    if is_band_filter:
        if cutoffs.ndim == 1:
            raise ValueError(
                f"For {filter_type} filters, cutoffs must have shape (batch, 2)"
            )
        if cutoffs.shape[-1] != 2:
            raise ValueError(
                f"For {filter_type} filters, cutoffs must have shape (batch, 2), "
                f"got {cutoffs.shape}"
            )
        batch_size = cutoffs.shape[0]
    else:
        if cutoffs.ndim == 2:
            # Squeeze if single-element second dimension
            if cutoffs.shape[-1] == 1:
                cutoffs = cutoffs.squeeze(-1)
            else:
                raise ValueError(
                    f"For {filter_type} filters, cutoffs must be 1D or (batch, 1), "
                    f"got {cutoffs.shape}"
                )
        batch_size = cutoffs.shape[0]

    # Design filters one by one, stacking results
    # This maintains gradient flow while being compatible with the existing API
    filters = []
    for i in range(batch_size):
        if is_band_filter:
            cutoff = cutoffs[i]
        else:
            cutoff = cutoffs[i]

        sos = butterworth_design(
            order=order,
            cutoff=cutoff,
            filter_type=filter_type,
            output="sos",
            sampling_frequency=sampling_frequency,
            dtype=dtype,
            device=device,
        )
        filters.append(sos)

    return torch.stack(filters, dim=0)


def batched_chebyshev_type_1_design(
    order: int,
    passband_ripple_db: float,
    cutoffs: Tensor,
    filter_type: Literal[
        "lowpass", "highpass", "bandpass", "bandstop"
    ] = "lowpass",
    output: Literal["sos"] = "sos",
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Design multiple Chebyshev Type I filters with different cutoff frequencies.

    Efficiently designs multiple filters in parallel.

    Parameters
    ----------
    order : int
        Filter order (same for all filters).
    passband_ripple_db : float
        Maximum ripple in the passband in decibels (same for all filters).
    cutoffs : Tensor
        Cutoff frequencies. Shape ``(batch,)`` for lowpass/highpass filters,
        or ``(batch, 2)`` for bandpass/bandstop filters.
    filter_type : {"lowpass", "highpass", "bandpass", "bandstop"}, optional
        Filter type. Default is "lowpass".
    output : {"sos"}, optional
        Output format. Currently only "sos" is supported for batched design.
    sampling_frequency : float, optional
        The sampling frequency of the digital system.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    filters : Tensor
        Batched filter coefficients. Shape ``(batch, n_sections, 6)``.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import batched_chebyshev_type_1_design
    >>> cutoffs = torch.linspace(0.1, 0.4, 5)
    >>> filters = batched_chebyshev_type_1_design(4, 1.0, cutoffs)
    >>> filters.shape
    torch.Size([5, 2, 6])
    """
    if output != "sos":
        raise ValueError(
            f"Batched design only supports 'sos' output, got '{output}'"
        )

    if cutoffs.numel() == 0:
        raise ValueError("cutoffs tensor cannot be empty")

    cutoffs = torch.atleast_1d(cutoffs)
    is_band_filter = filter_type in ("bandpass", "bandstop")

    if is_band_filter:
        if cutoffs.ndim == 1:
            raise ValueError(
                f"For {filter_type} filters, cutoffs must have shape (batch, 2)"
            )
        if cutoffs.shape[-1] != 2:
            raise ValueError(
                f"For {filter_type} filters, cutoffs must have shape (batch, 2)"
            )
        batch_size = cutoffs.shape[0]
    else:
        if cutoffs.ndim == 2:
            if cutoffs.shape[-1] == 1:
                cutoffs = cutoffs.squeeze(-1)
            else:
                raise ValueError(
                    f"For {filter_type} filters, cutoffs must be 1D"
                )
        batch_size = cutoffs.shape[0]

    filters = []
    for i in range(batch_size):
        cutoff = cutoffs[i]

        sos = chebyshev_type_1_design(
            order=order,
            cutoff=cutoff,
            passband_ripple_db=passband_ripple_db,
            filter_type=filter_type,
            output="sos",
            sampling_frequency=sampling_frequency,
            dtype=dtype,
            device=device,
        )
        filters.append(sos)

    return torch.stack(filters, dim=0)


def batched_chebyshev_type_2_design(
    order: int,
    stopband_attenuation_db: float,
    cutoffs: Tensor,
    filter_type: Literal[
        "lowpass", "highpass", "bandpass", "bandstop"
    ] = "lowpass",
    output: Literal["sos"] = "sos",
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Design multiple Chebyshev Type II filters with different cutoff frequencies.

    Efficiently designs multiple filters in parallel.

    Parameters
    ----------
    order : int
        Filter order (same for all filters).
    stopband_attenuation_db : float
        Minimum attenuation in stopband in decibels (same for all filters).
    cutoffs : Tensor
        Cutoff frequencies. Shape ``(batch,)`` for lowpass/highpass filters,
        or ``(batch, 2)`` for bandpass/bandstop filters.
    filter_type : {"lowpass", "highpass", "bandpass", "bandstop"}, optional
        Filter type. Default is "lowpass".
    output : {"sos"}, optional
        Output format. Currently only "sos" is supported for batched design.
    sampling_frequency : float, optional
        The sampling frequency of the digital system.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    filters : Tensor
        Batched filter coefficients. Shape ``(batch, n_sections, 6)``.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import batched_chebyshev_type_2_design
    >>> cutoffs = torch.linspace(0.1, 0.4, 5)
    >>> filters = batched_chebyshev_type_2_design(4, 40.0, cutoffs)
    >>> filters.shape
    torch.Size([5, 2, 6])
    """
    if output != "sos":
        raise ValueError(
            f"Batched design only supports 'sos' output, got '{output}'"
        )

    if cutoffs.numel() == 0:
        raise ValueError("cutoffs tensor cannot be empty")

    cutoffs = torch.atleast_1d(cutoffs)
    is_band_filter = filter_type in ("bandpass", "bandstop")

    if is_band_filter:
        if cutoffs.ndim == 1:
            raise ValueError(
                f"For {filter_type} filters, cutoffs must have shape (batch, 2)"
            )
        if cutoffs.shape[-1] != 2:
            raise ValueError(
                f"For {filter_type} filters, cutoffs must have shape (batch, 2)"
            )
        batch_size = cutoffs.shape[0]
    else:
        if cutoffs.ndim == 2:
            if cutoffs.shape[-1] == 1:
                cutoffs = cutoffs.squeeze(-1)
            else:
                raise ValueError(
                    f"For {filter_type} filters, cutoffs must be 1D"
                )
        batch_size = cutoffs.shape[0]

    filters = []
    for i in range(batch_size):
        cutoff = cutoffs[i]

        sos = chebyshev_type_2_design(
            order=order,
            cutoff=cutoff,
            stopband_attenuation_db=stopband_attenuation_db,
            filter_type=filter_type,
            output="sos",
            sampling_frequency=sampling_frequency,
            dtype=dtype,
            device=device,
        )
        filters.append(sos)

    return torch.stack(filters, dim=0)


def batched_firwin(
    num_taps: int,
    cutoffs: Tensor,
    filter_type: Literal[
        "lowpass", "highpass", "bandpass", "bandstop"
    ] = "lowpass",
    window: str = "hamming",
    pass_zero: Optional[bool] = None,
    scale: bool = True,
    sampling_frequency: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Design multiple FIR filters with different cutoff frequencies.

    Efficiently designs multiple filters in parallel.

    Parameters
    ----------
    num_taps : int
        Length of the filter (number of coefficients). Same for all filters.
    cutoffs : Tensor
        Cutoff frequencies. Shape ``(batch,)`` for lowpass/highpass filters,
        or ``(batch, 2)`` for bandpass/bandstop filters. Frequencies are
        expressed as a fraction of the Nyquist frequency (0 to 1).
    filter_type : {"lowpass", "highpass", "bandpass", "bandstop"}, optional
        Filter type. Default is "lowpass".
    window : str, optional
        Window function to use. Default is "hamming".
    pass_zero : bool, optional
        If True, the filter has gain at frequency 0 (DC).
    scale : bool, optional
        If True (default), scale for unity gain.
    sampling_frequency : float, optional
        The sampling frequency of the system.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    filters : Tensor
        Batched filter coefficients. Shape ``(batch, num_taps)``.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import batched_firwin
    >>> cutoffs = torch.linspace(0.1, 0.4, 10)
    >>> filters = batched_firwin(51, cutoffs)
    >>> filters.shape
    torch.Size([10, 51])
    """
    if cutoffs.numel() == 0:
        raise ValueError("cutoffs tensor cannot be empty")

    cutoffs = torch.atleast_1d(cutoffs)
    is_band_filter = filter_type in ("bandpass", "bandstop")

    if is_band_filter:
        if cutoffs.ndim == 1:
            raise ValueError(
                f"For {filter_type} filters, cutoffs must have shape (batch, 2)"
            )
        if cutoffs.shape[-1] != 2:
            raise ValueError(
                f"For {filter_type} filters, cutoffs must have shape (batch, 2)"
            )
        batch_size = cutoffs.shape[0]
    else:
        if cutoffs.ndim == 2:
            if cutoffs.shape[-1] == 1:
                cutoffs = cutoffs.squeeze(-1)
            else:
                raise ValueError(
                    f"For {filter_type} filters, cutoffs must be 1D"
                )
        batch_size = cutoffs.shape[0]

    filters = []
    for i in range(batch_size):
        cutoff = cutoffs[i]

        h = firwin(
            num_taps=num_taps,
            cutoff=cutoff,
            filter_type=filter_type,
            window=window,
            pass_zero=pass_zero,
            scale=scale,
            sampling_frequency=sampling_frequency,
            dtype=dtype,
            device=device,
        )
        filters.append(h)

    return torch.stack(filters, dim=0)


def batched_filter_apply(
    filters: Tensor,
    x: Tensor,
    filter_format: Literal["sos", "fir"] = "sos",
    broadcast: bool = True,
    axis: int = -1,
) -> Tensor:
    """
    Apply multiple filters to a signal or batched signals.

    Parameters
    ----------
    filters : Tensor
        Batched filter coefficients from batched design functions.
        For SOS format: shape ``(batch, n_sections, 6)``.
        For FIR format: shape ``(batch, num_taps)``.
    x : Tensor
        Input signal(s). Can be 1D or batched.
    filter_format : {"sos", "fir"}, optional
        Format of the filter coefficients. Default is "sos".
    broadcast : bool, optional
        If True (default), apply all filters to the same signal (x is
        broadcasted). If False, apply filters element-wise to batched
        signals (filters and x must have the same batch dimension).
    axis : int, optional
        Axis along which to filter. Default is -1 (last axis).

    Returns
    -------
    y : Tensor
        Filtered signals. Shape depends on broadcast mode:
        - If broadcast=True: ``(n_filters, *signal_shape)``
        - If broadcast=False: ``(batch, *signal_shape)``

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import (
    ...     batched_butterworth_design, batched_filter_apply
    ... )
    >>> cutoffs = torch.tensor([0.1, 0.2, 0.3])
    >>> sos = batched_butterworth_design(4, cutoffs)
    >>> x = torch.randn(100, dtype=torch.float64)
    >>> y = batched_filter_apply(sos, x)
    >>> y.shape
    torch.Size([3, 100])

    Apply different filters to different signals:

    >>> x_batch = torch.randn(3, 100, dtype=torch.float64)
    >>> y = batched_filter_apply(sos, x_batch, broadcast=False)
    >>> y.shape
    torch.Size([3, 100])
    """
    n_filters = filters.shape[0]

    if filter_format == "sos":
        # SOS format: shape (batch, n_sections, 6)
        if filters.ndim != 3 or filters.shape[2] != 6:
            raise ValueError(
                f"SOS filters must have shape (batch, n_sections, 6), "
                f"got {filters.shape}"
            )

        if broadcast:
            # Apply all filters to the same signal
            outputs = []
            for i in range(n_filters):
                y = sosfilt(filters[i], x, axis=axis)
                outputs.append(y)
            return torch.stack(outputs, dim=0)
        else:
            # Apply filters element-wise to batched signals
            if x.shape[0] != n_filters:
                raise ValueError(
                    f"When broadcast=False, x batch dim must match filters: "
                    f"{x.shape[0]} != {n_filters}"
                )
            outputs = []
            for i in range(n_filters):
                y = sosfilt(filters[i], x[i], axis=axis)
                outputs.append(y)
            return torch.stack(outputs, dim=0)

    elif filter_format == "fir":
        # FIR format: shape (batch, num_taps)
        if filters.ndim != 2:
            raise ValueError(
                f"FIR filters must have shape (batch, num_taps), "
                f"got {filters.shape}"
            )

        if broadcast:
            outputs = []
            for i in range(n_filters):
                b = filters[i]
                a = torch.ones(1, dtype=b.dtype, device=b.device)
                y = lfilter(b, a, x, axis=axis)
                outputs.append(y)
            return torch.stack(outputs, dim=0)
        else:
            if x.shape[0] != n_filters:
                raise ValueError(
                    f"When broadcast=False, x batch dim must match filters: "
                    f"{x.shape[0]} != {n_filters}"
                )
            outputs = []
            for i in range(n_filters):
                b = filters[i]
                a = torch.ones(1, dtype=b.dtype, device=b.device)
                y = lfilter(b, a, x[i], axis=axis)
                outputs.append(y)
            return torch.stack(outputs, dim=0)

    else:
        raise ValueError(
            f"filter_format must be 'sos' or 'fir', got '{filter_format}'"
        )

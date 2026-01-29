"""Discrete wavelet transform implementation."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401

from ._wavelets import get_wavelet_filters

# Padding mode mapping: Python string -> C++ int
_PADDING_MODE_MAP = {
    "symmetric": 0,
    "reflect": 1,
    "periodic": 2,
    "zero": 3,
}


def _dwt_coeff_len(input_len: int, filter_len: int, mode: int) -> int:
    """Compute DWT output length matching PyWavelets behavior."""
    if mode == 2:  # periodic
        return (input_len + 1) // 2
    else:
        # symmetric, reflect, zero: include boundary effects
        return (input_len + filter_len - 1) // 2


def _compute_coeff_lengths(
    input_length: int, filter_len: int, levels: int, mode: int = 0
) -> list[int]:
    """Compute coefficient lengths for each DWT level.

    Returns a list where coeff_lens[i] is the coefficient length at level i+1.
    """
    lengths = []
    current_len = input_length
    for _ in range(levels):
        coeff_len = _dwt_coeff_len(current_len, filter_len, mode)
        lengths.append(coeff_len)
        current_len = coeff_len
    return lengths


def _unpack_coefficients(
    packed: Tensor,
    input_length: int,
    filter_len: int,
    levels: int,
    mode: int = 0,
) -> tuple[Tensor, list[Tensor]]:
    """Unpack coefficients from packed format to (approx, [details]).

    Packed format: [cA_n | cD_n | cD_{n-1} | ... | cD_1]
    Returns: (approx, [d1, d2, ..., dn]) where d1 is finest (level 1)
    """
    coeff_lens = _compute_coeff_lengths(input_length, filter_len, levels, mode)

    # Approximation coefficients (at the coarsest level)
    approx_len = coeff_lens[-1]
    approx = packed.narrow(-1, 0, approx_len)

    # Detail coefficients are stored as cD_n, cD_{n-1}, ..., cD_1
    # We need to return them as [d1, d2, ..., dn] (finest to coarsest)
    details = []
    offset = approx_len

    # coeff_lens[i] is the length at level i+1
    # Details in packed: cD_n (level n), cD_{n-1} (level n-1), ..., cD_1 (level 1)
    # So we read in order of levels-1, levels-2, ..., 0
    for i in range(levels - 1, -1, -1):
        detail_len = coeff_lens[i]
        detail = packed.narrow(-1, offset, detail_len)
        details.append(detail)
        offset += detail_len

    # Reverse to get [d1, d2, ..., dn] (finest to coarsest)
    details.reverse()

    return approx, details


def discrete_wavelet_transform(
    input: Tensor,
    wavelet: str = "haar",
    *,
    level: int = 1,
    dim: int = -1,
    padding_mode: Literal[
        "symmetric", "reflect", "periodic", "zero"
    ] = "symmetric",
) -> tuple[Tensor, list[Tensor]]:
    r"""Compute the discrete wavelet transform (DWT) of a signal.

    The DWT decomposes a signal into approximation and detail coefficients
    at multiple scales using filter bank convolution and downsampling.

    At each level, the signal is convolved with lowpass and highpass filters
    and then downsampled by a factor of 2:

    .. math::
        a_j[n] = \sum_k h[k-2n] \cdot a_{j-1}[k]

        d_j[n] = \sum_k g[k-2n] \cdot a_{j-1}[k]

    where :math:`h` is the lowpass (scaling) filter, :math:`g` is the highpass
    (wavelet) filter, :math:`a_j` are the approximation coefficients at level
    :math:`j`, and :math:`d_j` are the detail coefficients.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. The transform is computed along ``dim``.
    wavelet : str, optional
        Name of the wavelet to use. Supported wavelets:

        - ``"haar"``: Haar wavelet (simplest, 2 coefficients)
        - ``"db1"``: Daubechies-1 (same as Haar)
        - ``"db2"``: Daubechies-2 (4 coefficients, smoother)
        - ``"db3"``: Daubechies-3 (6 coefficients)
        - ``"db4"``: Daubechies-4 (8 coefficients)

        Default: ``"haar"``.
    level : int, optional
        Number of decomposition levels. Must be >= 1.
        The maximum useful level depends on the signal length and filter length.
        If ``level`` exceeds the maximum, it will be clamped.
        Default: ``1``.
    dim : int, optional
        The dimension along which to compute the transform.
        Default: ``-1`` (last dimension).
    padding_mode : str, optional
        Padding mode for boundary handling. One of:

        - ``"symmetric"``: Symmetric extension (half-sample symmetric).
        - ``"reflect"``: Reflect at boundaries (whole-sample symmetric).
        - ``"periodic"``: Periodic (circular) extension.
        - ``"zero"``: Zero padding.

        Default: ``"symmetric"``.

    Returns
    -------
    approx : Tensor
        Final approximation coefficients from the last decomposition level.
        Shape is ``(..., ceil(N / 2^level), ...)`` where N is the input size
        along ``dim``.
    details : list of Tensor
        List of detail coefficients from each level, ordered from finest
        (level 1) to coarsest (level L). The i-th element has shape
        ``(..., ceil(N / 2^i), ...)``.

    Raises
    ------
    ValueError
        If ``wavelet`` is not a recognized wavelet name.
        If ``level`` is less than 1.

    Examples
    --------
    Single-level Haar wavelet transform:

    >>> x = torch.randn(128)
    >>> approx, details = discrete_wavelet_transform(x, wavelet="haar")
    >>> approx.shape
    torch.Size([64])
    >>> len(details)
    1
    >>> details[0].shape
    torch.Size([64])

    Multi-level decomposition:

    >>> x = torch.randn(256)
    >>> approx, details = discrete_wavelet_transform(x, wavelet="db2", level=3)
    >>> approx.shape
    torch.Size([32])
    >>> [d.shape for d in details]
    [torch.Size([128]), torch.Size([64]), torch.Size([32])]

    Batched input:

    >>> x = torch.randn(4, 128)
    >>> approx, details = discrete_wavelet_transform(x, wavelet="haar")
    >>> approx.shape
    torch.Size([4, 64])

    Notes
    -----
    **Perfect Reconstruction:**

    The DWT is designed for perfect reconstruction. Using
    ``inverse_discrete_wavelet_transform`` with the same wavelet will
    recover the original signal (up to numerical precision).

    **Energy Preservation:**

    For orthogonal wavelets (Haar, Daubechies), the transform preserves
    signal energy:

    .. math::
        \|x\|^2 = \|a_L\|^2 + \sum_{j=1}^{L} \|d_j\|^2

    **Filter Bank Interpretation:**

    The DWT can be viewed as a critically-sampled filter bank. The lowpass
    and highpass filters form a perfect reconstruction filter bank.

    See Also
    --------
    inverse_discrete_wavelet_transform : The inverse DWT.
    short_time_fourier_transform : Time-frequency analysis via STFT.
    """
    # Validate level
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}")

    # Get wavelet filters (this also validates the wavelet name)
    dec_lo, dec_hi, _, _ = get_wavelet_filters(
        wavelet, dtype=input.dtype, device=input.device
    )

    filter_len = dec_lo.shape[0]

    # Normalize dimension
    ndim = input.ndim
    normalized_dim = dim if dim >= 0 else dim + ndim

    # Move transform dimension to last position
    if normalized_dim != ndim - 1:
        x = input.movedim(normalized_dim, -1)
    else:
        x = input

    # Compute maximum valid level based on signal length
    signal_len = x.shape[-1]
    # Max level: need at least filter_len samples to decompose
    if signal_len >= filter_len:
        max_level = int(math.log2(signal_len / (filter_len - 1)))
        max_level = max(max_level, 1)
    else:
        max_level = 1

    # Clamp level to maximum
    actual_level = min(level, max_level)

    # Convert padding mode to int
    mode_int = _PADDING_MODE_MAP[padding_mode]

    # Call C++ backend
    packed = torch.ops.torchscience.discrete_wavelet_transform(
        x, dec_lo, dec_hi, actual_level, mode_int
    )

    # Unpack coefficients to (approx, [details]) format
    approx, details = _unpack_coefficients(
        packed, signal_len, filter_len, actual_level, mode_int
    )

    # Move dimension back if needed
    if normalized_dim != ndim - 1:
        approx = approx.movedim(-1, normalized_dim)
        details = [d.movedim(-1, normalized_dim) for d in details]

    return approx, details

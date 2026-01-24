"""Inverse discrete wavelet transform implementation."""

from __future__ import annotations

from typing import Literal

import torch.nn.functional as F
from torch import Tensor

from ._wavelets import get_wavelet_filters


def _idwt_single_level(
    approx: Tensor,
    detail: Tensor,
    rec_lo: Tensor,
    rec_hi: Tensor,
    output_len: int,
) -> Tensor:
    """Perform single-level IDWT reconstruction.

    Uses transposed convolution (conv_transpose1d) for upsampling and filtering,
    which is the correct way to implement synthesis filter banks for perfect
    reconstruction.

    Parameters
    ----------
    approx : Tensor
        Approximation coefficients of shape (..., N_coeff).
    detail : Tensor
        Detail coefficients of shape (..., N_coeff).
    rec_lo : Tensor
        Reconstruction lowpass filter of shape (filter_len,).
    rec_hi : Tensor
        Reconstruction highpass filter of shape (filter_len,).
    output_len : int
        Expected output signal length.

    Returns
    -------
    Tensor
        Reconstructed signal of shape (..., output_len).
    """
    # Prepare inputs for conv_transpose1d: need shape (batch, channels, length)
    original_shape = approx.shape
    coeff_len = original_shape[-1]

    # Flatten all batch dimensions
    if approx.ndim == 1:
        approx_conv = approx.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        detail_conv = detail.unsqueeze(0).unsqueeze(0)
        batch_shape = ()
    else:
        batch_numel = 1
        for s in original_shape[:-1]:
            batch_numel *= s
        approx_conv = approx.reshape(
            batch_numel, 1, coeff_len
        )  # (batch, 1, N)
        detail_conv = detail.reshape(batch_numel, 1, coeff_len)
        batch_shape = original_shape[:-1]

    # Prepare filters for conv_transpose1d: shape (in_channels, out_channels/groups, kernel_size)
    # Note: conv_transpose1d has shape (in_channels, out_channels, kernel_size)
    rec_lo_kernel = rec_lo.reshape(1, 1, -1)
    rec_hi_kernel = rec_hi.reshape(1, 1, -1)

    # Use conv_transpose1d with stride=2 (which does upsampling + convolution)
    # This is the correct way to implement the synthesis filter bank
    approx_rec = F.conv_transpose1d(approx_conv, rec_lo_kernel, stride=2)
    detail_rec = F.conv_transpose1d(detail_conv, rec_hi_kernel, stride=2)

    # Add the lowpass and highpass reconstructions
    reconstructed = approx_rec + detail_rec

    # Trim to output length if needed
    # The output of conv_transpose1d with input N, filter L, stride 2 is:
    # (N - 1) * 2 + L
    current_len = reconstructed.shape[-1]
    if current_len > output_len:
        # Trim from the end
        reconstructed = reconstructed[:, :, :output_len]
    elif current_len < output_len:
        # Pad at the end if needed
        pad_needed = output_len - current_len
        reconstructed = F.pad(reconstructed, (0, pad_needed))

    # Reshape back to original batch shape
    if len(batch_shape) == 0:
        reconstructed = reconstructed.squeeze(0).squeeze(0)  # (output_len,)
    else:
        reconstructed = reconstructed.squeeze(1).reshape(*batch_shape, -1)

    return reconstructed


def inverse_discrete_wavelet_transform(
    coeffs: tuple[Tensor, list[Tensor]],
    wavelet: str = "haar",
    *,
    dim: int = -1,
    padding_mode: Literal[
        "symmetric", "reflect", "periodic", "zero"
    ] = "symmetric",
) -> Tensor:
    r"""Reconstruct a signal from its discrete wavelet transform coefficients.

    The inverse DWT reconstructs the original signal from approximation and
    detail coefficients using synthesis filter bank convolution and upsampling.

    At each level, the coefficients are upsampled by 2 and convolved with
    reconstruction filters, then summed:

    .. math::
        a_{j-1}[n] = \sum_k \tilde{h}[n-2k] \cdot a_j[k] +
                     \sum_k \tilde{g}[n-2k] \cdot d_j[k]

    where :math:`\tilde{h}` is the reconstruction lowpass filter,
    :math:`\tilde{g}` is the reconstruction highpass filter, :math:`a_j` are
    the approximation coefficients at level :math:`j`, and :math:`d_j` are
    the detail coefficients.

    Parameters
    ----------
    coeffs : tuple of (Tensor, list of Tensor)
        DWT coefficients as returned by ``discrete_wavelet_transform``:

        - ``coeffs[0]``: Final approximation coefficients (Tensor)
        - ``coeffs[1]``: List of detail coefficients from each level,
          ordered from finest (level 1) to coarsest (level L)

    wavelet : str, optional
        Name of the wavelet to use. Must match the wavelet used for
        decomposition. Supported wavelets:

        - ``"haar"``: Haar wavelet (simplest, 2 coefficients)
        - ``"db1"``: Daubechies-1 (same as Haar)
        - ``"db2"``: Daubechies-2 (4 coefficients, smoother)
        - ``"db3"``: Daubechies-3 (6 coefficients)
        - ``"db4"``: Daubechies-4 (8 coefficients)

        Default: ``"haar"``.
    dim : int, optional
        The dimension along which to compute the inverse transform.
        Must match the ``dim`` used in ``discrete_wavelet_transform``.
        Default: ``-1`` (last dimension).
    padding_mode : str, optional
        Padding mode for boundary handling. Must match the ``padding_mode``
        used in ``discrete_wavelet_transform``. One of:

        - ``"symmetric"``: Symmetric extension (half-sample symmetric).
        - ``"reflect"``: Reflect at boundaries (whole-sample symmetric).
        - ``"periodic"``: Periodic (circular) extension.
        - ``"zero"``: Zero padding.

        Default: ``"symmetric"``.

    Returns
    -------
    Tensor
        Reconstructed signal with the same shape as the original input
        to ``discrete_wavelet_transform``.

    Raises
    ------
    ValueError
        If ``wavelet`` is not a recognized wavelet name.

    Examples
    --------
    Reconstruct from single-level DWT:

    >>> x = torch.randn(128)
    >>> approx, details = discrete_wavelet_transform(x, wavelet="haar")
    >>> x_rec = inverse_discrete_wavelet_transform((approx, details), wavelet="haar")
    >>> torch.allclose(x, x_rec, atol=1e-5)
    True

    Reconstruct from multi-level DWT:

    >>> x = torch.randn(256)
    >>> approx, details = discrete_wavelet_transform(x, wavelet="db2", level=3)
    >>> x_rec = inverse_discrete_wavelet_transform((approx, details), wavelet="db2")
    >>> torch.allclose(x, x_rec, atol=1e-4)
    True

    Batched reconstruction:

    >>> x = torch.randn(4, 128)
    >>> approx, details = discrete_wavelet_transform(x, wavelet="haar")
    >>> x_rec = inverse_discrete_wavelet_transform((approx, details), wavelet="haar")
    >>> x_rec.shape
    torch.Size([4, 128])

    Notes
    -----
    **Perfect Reconstruction:**

    Using the same wavelet and padding mode as the forward transform,
    the inverse transform recovers the original signal up to numerical
    precision (typically 1e-5 to 1e-4 relative error).

    **Filter Bank Interpretation:**

    The IDWT uses synthesis filters that form a perfect reconstruction
    pair with the analysis filters used in the DWT. For orthogonal
    wavelets, the synthesis filters are simply time-reversed versions
    of the analysis filters.

    See Also
    --------
    discrete_wavelet_transform : The forward DWT.
    """
    approx, details = coeffs

    # Get wavelet filters (this also validates the wavelet name)
    _, _, rec_lo, rec_hi = get_wavelet_filters(
        wavelet, dtype=approx.dtype, device=approx.device
    )

    # Number of levels
    num_levels = len(details)

    if num_levels == 0:
        # No detail coefficients, just return approx
        return approx

    # Normalize dimension
    ndim = approx.ndim
    normalized_dim = dim if dim >= 0 else dim + ndim

    # Move transform dimension to last position
    if normalized_dim != ndim - 1:
        approx = approx.movedim(normalized_dim, -1)
        details = [d.movedim(normalized_dim, -1) for d in details]

    # Reconstruct from coarsest to finest level
    # Details are ordered [d1, d2, ..., dn] (finest to coarsest)
    # We need to iterate from coarsest to finest: [dn, d_{n-1}, ..., d1]
    reconstructed = approx

    for i in range(num_levels - 1, -1, -1):
        detail = details[i]

        # Compute expected output length from this level
        # The output length should be 2 * coeff_len for perfect reconstruction
        coeff_len = detail.shape[-1]
        output_len = coeff_len * 2

        reconstructed = _idwt_single_level(
            reconstructed, detail, rec_lo, rec_hi, output_len
        )

    # Move dimension back if needed
    if normalized_dim != ndim - 1:
        reconstructed = reconstructed.movedim(-1, normalized_dim)

    return reconstructed

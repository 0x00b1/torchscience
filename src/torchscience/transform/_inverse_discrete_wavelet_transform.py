"""Inverse discrete wavelet transform implementation."""

from __future__ import annotations

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


def _compute_coeff_lengths(input_length: int, levels: int) -> list[int]:
    """Compute coefficient lengths for each DWT level."""
    lengths = []
    current_len = input_length
    for _ in range(levels):
        coeff_len = (current_len + 1) // 2
        lengths.append(coeff_len)
        current_len = coeff_len
    return lengths


def _pack_coefficients(approx: Tensor, details: list[Tensor]) -> Tensor:
    """Pack (approx, [details]) into packed format.

    Input: (approx, [d1, d2, ..., dn]) where d1 is finest (level 1)
    Packed format: [cA_n | cD_n | cD_{n-1} | ... | cD_1]
    """
    # Details are [d1, d2, ..., dn] (finest to coarsest)
    # Pack as [approx, dn, d_{n-1}, ..., d1]
    all_coeffs = [approx]
    for detail in reversed(details):
        all_coeffs.append(detail)
    return torch.cat(all_coeffs, dim=-1)


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

    # Compute output length: work backwards from finest detail
    # details[0] is from level 1, which was (input_len + 1) // 2
    # So input_len = details[0].size(-1) * 2 or details[0].size(-1) * 2 - 1
    # We use the even case (details[0].size(-1) * 2) by default
    output_length = details[0].shape[-1] * 2

    # Convert padding mode to int
    mode_int = _PADDING_MODE_MAP[padding_mode]

    # Pack coefficients for C++ backend
    packed = _pack_coefficients(approx, details)

    # Call C++ backend
    reconstructed = torch.ops.torchscience.inverse_discrete_wavelet_transform(
        packed, rec_lo, rec_hi, num_levels, mode_int, output_length
    )

    # Move dimension back if needed
    if normalized_dim != ndim - 1:
        reconstructed = reconstructed.movedim(-1, normalized_dim)

    return reconstructed

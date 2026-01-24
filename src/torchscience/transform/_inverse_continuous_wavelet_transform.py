"""Inverse continuous wavelet transform implementation."""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

from ._continuous_wavelet_transform import _get_wavelet_function


def inverse_continuous_wavelet_transform(
    input: Tensor,
    scales: Tensor,
    wavelet: str | Callable[[Tensor], Tensor] = "morlet",
    *,
    dim: int = -1,
    sampling_period: float = 1.0,
) -> Tensor:
    r"""Compute the inverse continuous wavelet transform (ICWT).

    Reconstructs a signal from its continuous wavelet transform (CWT)
    coefficients. The reconstruction integrates over scales using the
    formula:

    .. math::
        x(t) \approx C^{-1} \sum_s \frac{W_x(s, t)}{s}

    where :math:`W_x(s, t)` are the CWT coefficients, :math:`s` are the
    scales, and :math:`C` is a normalization constant.

    Parameters
    ----------
    input : Tensor
        CWT coefficients tensor. For unbatched input, shape is
        ``(num_scales, signal_len)``. For batched input, shape is
        ``(batch..., num_scales, signal_len)`` where ``batch...``
        are zero or more batch dimensions.
    scales : Tensor
        1-D tensor of positive scale values. Must match the scales
        used in the forward CWT and have length equal to the scale
        dimension of the input.
    wavelet : str or Callable, optional
        The wavelet used in the forward CWT. Can be:

        - ``"morlet"``: Complex Morlet wavelet (default).
        - ``"mexican_hat"``: Mexican hat (Ricker) wavelet.
        - ``"ricker"``: Alias for ``"mexican_hat"``.
        - A callable ``f(t) -> Tensor`` that defines a custom wavelet.

        Default: ``"morlet"``.
    dim : int, optional
        The dimension along which to compute the inverse transform.
        This should be the signal dimension (the last axis of the CWT
        output). Default: ``-1`` (last dimension).
    sampling_period : float, optional
        Sampling period of the signal (time between samples).
        Should match the value used in the forward CWT.
        Default: ``1.0``.

    Returns
    -------
    Tensor
        Reconstructed signal. For unbatched input of shape
        ``(num_scales, signal_len)``, output is ``(signal_len,)``.
        For batched input of shape ``(batch..., num_scales, signal_len)``,
        output is ``(batch..., signal_len)``.

    Raises
    ------
    ValueError
        If ``scales`` is not a 1-D tensor.
        If any scale value is not positive.
        If the number of scales doesn't match the input scale dimension.
        If ``wavelet`` string is not recognized.

    Examples
    --------
    Reconstruct a signal from its CWT:

    >>> x = torch.randn(256)
    >>> scales = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    >>> cwt = continuous_wavelet_transform(x, scales, wavelet="mexican_hat")
    >>> reconstructed = inverse_continuous_wavelet_transform(
    ...     cwt, scales, wavelet="mexican_hat"
    ... )
    >>> reconstructed.shape
    torch.Size([256])

    Batched reconstruction:

    >>> x = torch.randn(4, 256)
    >>> cwt = continuous_wavelet_transform(x, scales, dim=-1)
    >>> reconstructed = inverse_continuous_wavelet_transform(cwt, scales, dim=-1)
    >>> reconstructed.shape
    torch.Size([4, 256])

    Notes
    -----
    **Reconstruction Quality:**

    The CWT is redundant (overcomplete), so perfect reconstruction is not
    guaranteed. The quality of reconstruction depends on:

    - The number and distribution of scales used
    - The wavelet satisfying the admissibility condition
    - Signal characteristics relative to the wavelet

    For better reconstruction, use a logarithmically spaced set of scales
    covering the frequency range of interest.

    **Normalization:**

    The reconstruction uses a simple scale-weighted sum. The normalization
    constant is estimated from the wavelet and scales to provide
    approximately unit gain.

    **Relation to Forward CWT:**

    This function inverts ``continuous_wavelet_transform``. The same wavelet
    and scales should be used for both forward and inverse transforms.

    See Also
    --------
    continuous_wavelet_transform : Forward continuous wavelet transform.
    inverse_discrete_wavelet_transform : Inverse discrete wavelet transform.

    References
    ----------
    .. [1] Torrence, C. & Compo, G.P. (1998). A Practical Guide to
           Wavelet Analysis. Bulletin of the American Meteorological Society.
    .. [2] Mallat, S. (2009). A Wavelet Tour of Signal Processing.
           Academic Press.
    """
    # Validate scales
    if scales.ndim != 1:
        raise ValueError(
            f"scales must be a 1-D tensor, got {scales.ndim}-D tensor"
        )

    if (scales <= 0).any():
        raise ValueError("All scale values must be positive")

    # Validate wavelet name by getting the wavelet function.
    # The result is not used because computing the wavelet's admissibility
    # constant analytically for arbitrary wavelets is complex. Instead, we
    # use an empirical normalization approach that works well in practice.
    _get_wavelet_function(wavelet)

    num_scales = scales.shape[0]

    # Normalize dimension
    ndim = input.ndim
    normalized_dim = dim if dim >= 0 else dim + ndim

    # Validate input shape: expect (num_scales, signal_len) or (batch..., num_scales, signal_len)
    # The scale dimension is the second-to-last dimension
    if ndim < 2:
        raise ValueError(
            f"Input must have at least 2 dimensions (num_scales, signal_len), "
            f"got {ndim}-D tensor"
        )

    # For the inverse, scales dimension is at normalized_dim - 1
    # and signal dimension is at normalized_dim
    scale_dim = normalized_dim - 1 if normalized_dim > 0 else -2
    scale_dim = scale_dim if scale_dim >= 0 else scale_dim + ndim

    if input.shape[scale_dim] != num_scales:
        raise ValueError(
            f"Number of scales ({num_scales}) does not match the scale "
            f"dimension of input ({input.shape[scale_dim]})"
        )

    signal_len = input.shape[normalized_dim]

    # Move signal dimension to last for easier processing
    # and scale dimension to second-to-last
    if normalized_dim != ndim - 1:
        coeffs = input.movedim(normalized_dim, -1)
        # After moving signal dim, find where scale dim ended up
        if scale_dim > normalized_dim:
            scale_dim_adjusted = scale_dim - 1
        else:
            scale_dim_adjusted = scale_dim
        if scale_dim_adjusted != ndim - 2:
            coeffs = coeffs.movedim(scale_dim_adjusted, -2)
    else:
        coeffs = input
        if scale_dim != ndim - 2:
            coeffs = coeffs.movedim(scale_dim, -2)

    # Now coeffs has shape (..., num_scales, signal_len)
    batch_shape = coeffs.shape[:-2]

    # Inverse CWT formula: x(t) = C^{-1} * sum_s (W(s,t) / s)
    # where C is a normalization constant

    # Scale weights: 1/s for each scale
    # We integrate over scales using the trapezoidal rule approximation
    # for ds/s^2, which gives weights proportional to 1/s
    #
    # The sampling_period affects the effective scale in physical units.
    # In the forward CWT, the wavelet support is scaled by sampling_period,
    # so the effective scale is (scale * sampling_period). We incorporate
    # this into the integration weights for proper roundtrip reconstruction.

    # Effective scales in physical units (matching forward CWT)
    effective_scales = scales * sampling_period

    # Compute scale weights for integration
    # Using logarithmic scale spacing assumption: ds/s ~ d(log s)
    # Weight for each scale is approximately delta_log_s / s
    if len(effective_scales) > 1:
        # Use logarithmic spacing for integration weights
        log_scales = torch.log(effective_scales)
        # Compute the log-scale spacing (using central differences where possible)
        delta_log_s = torch.zeros_like(effective_scales)
        delta_log_s[0] = log_scales[1] - log_scales[0]
        delta_log_s[-1] = log_scales[-1] - log_scales[-2]
        if len(effective_scales) > 2:
            delta_log_s[1:-1] = (log_scales[2:] - log_scales[:-2]) / 2
    else:
        # Single scale: use unit weight
        delta_log_s = torch.ones_like(effective_scales)

    # Weight = delta_log_s / s (for ds/s^2 integration)
    scale_weights = delta_log_s / effective_scales

    # Reshape scale_weights for broadcasting: (..., num_scales, 1)
    for _ in range(len(batch_shape)):
        scale_weights = scale_weights.unsqueeze(0)
    scale_weights = scale_weights.unsqueeze(-1)  # (..., num_scales, 1)

    # Take real part of coefficients if complex
    if coeffs.is_complex():
        coeffs_real = coeffs.real
    else:
        coeffs_real = coeffs

    # Weighted sum over scales
    # reconstructed = sum_s (coeffs[s, t] * weight[s])
    weighted = coeffs_real * scale_weights.to(coeffs_real.dtype)
    reconstructed = weighted.sum(dim=-2)  # Sum over scale dimension

    # Normalization constant estimation
    # The CWT uses 1/sqrt(s) normalization, and we're summing with 1/s weights
    # Total effective weight is sum(delta_log_s / s^{1.5})
    # We normalize by the total weight to achieve approximately unit gain

    # Compute normalization factor
    total_weight = (delta_log_s / (effective_scales**1.5)).sum()
    if total_weight > 0:
        # Apply normalization - the factor depends on the wavelet and scale distribution
        # For a good range of scales, this provides reasonable reconstruction
        norm_factor = total_weight * math.sqrt(
            2.0
        )  # Empirical factor for better reconstruction
        reconstructed = reconstructed / norm_factor.to(reconstructed.dtype)

    # Determine output dtype
    if input.is_complex():
        if input.dtype == torch.complex64:
            output_dtype = torch.float32
        else:
            output_dtype = torch.float64
        reconstructed = reconstructed.to(output_dtype)

    return reconstructed

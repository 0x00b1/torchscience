"""Continuous wavelet transform implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor

if TYPE_CHECKING:
    pass


def morlet_wavelet(t: Tensor, omega0: float = 5.0) -> Tensor:
    """Morlet wavelet (complex-valued).

    The Morlet wavelet is defined as:

    .. math::
        \\psi(t) = e^{i \\omega_0 t} e^{-t^2 / 2}

    Parameters
    ----------
    t : Tensor
        Time points at which to evaluate the wavelet.
    omega0 : float, optional
        Center frequency of the wavelet. Default: ``5.0``.
        Higher values give better frequency resolution but worse time resolution.

    Returns
    -------
    Tensor
        Complex-valued wavelet at the given time points.

    Notes
    -----
    The admissibility condition requires :math:`\\omega_0 \\geq 5` for the
    Morlet wavelet to be a valid wavelet (zero mean approximation).
    """
    # Convert to complex dtype for the output
    if t.dtype == torch.float32:
        complex_dtype = torch.complex64
    elif t.dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        complex_dtype = torch.complex64

    # e^{i*omega0*t} * e^{-t^2/2}
    oscillation = torch.exp(1j * omega0 * t.to(complex_dtype))
    envelope = torch.exp(-(t**2) / 2)
    return oscillation * envelope


def mexican_hat_wavelet(t: Tensor) -> Tensor:
    """Mexican hat (Ricker) wavelet (real-valued).

    The Mexican hat wavelet is the negative normalized second derivative
    of a Gaussian:

    .. math::
        \\psi(t) = (1 - t^2) e^{-t^2 / 2}

    Parameters
    ----------
    t : Tensor
        Time points at which to evaluate the wavelet.

    Returns
    -------
    Tensor
        Real-valued wavelet at the given time points.

    Notes
    -----
    This wavelet is also known as the Ricker wavelet in geophysics.
    It has zero mean (admissibility condition) and is symmetric.
    """
    return (1 - t**2) * torch.exp(-(t**2) / 2)


# Registry of built-in wavelets
_WAVELET_REGISTRY: dict[str, Callable[[Tensor], Tensor]] = {
    "morlet": morlet_wavelet,
    "mexican_hat": mexican_hat_wavelet,
    "ricker": mexican_hat_wavelet,  # Alias for mexican_hat
}


def _get_wavelet_function(
    wavelet: str | Callable[[Tensor], Tensor],
) -> Callable[[Tensor], Tensor]:
    """Get wavelet function from string name or callable.

    Parameters
    ----------
    wavelet : str or Callable
        Either a string naming a built-in wavelet or a callable
        that takes a time tensor and returns wavelet values.

    Returns
    -------
    Callable
        The wavelet function.

    Raises
    ------
    ValueError
        If wavelet string is not recognized.
    """
    if callable(wavelet):
        return wavelet

    wavelet_lower = wavelet.lower()
    if wavelet_lower not in _WAVELET_REGISTRY:
        available = sorted(
            set(_WAVELET_REGISTRY.keys()) - {"ricker"}
        )  # Don't show alias
        raise ValueError(
            f"Unknown wavelet '{wavelet}'. "
            f"Available wavelets: {available}. "
            "You can also pass a custom wavelet function."
        )

    return _WAVELET_REGISTRY[wavelet_lower]


def _fft_convolve(
    signal: Tensor,
    kernel: Tensor,
) -> Tensor:
    """Convolve signal with kernel using FFT.

    The convolution is performed in the frequency domain for efficiency.
    Output size matches the signal size (same-mode convolution).

    Parameters
    ----------
    signal : Tensor
        Input signal of shape (..., N).
    kernel : Tensor
        Convolution kernel of shape (M,).

    Returns
    -------
    Tensor
        Convolution result of shape (..., N).
    """
    signal_len = signal.shape[-1]
    kernel_len = kernel.shape[-1]

    # Determine output length for full convolution
    full_len = signal_len + kernel_len - 1

    # Pad to power of 2 for FFT efficiency
    fft_len = 1 << (full_len - 1).bit_length()

    # Compute FFT of signal and kernel
    signal_fft = torch.fft.fft(signal, n=fft_len, dim=-1)
    kernel_fft = torch.fft.fft(kernel, n=fft_len)

    # Multiply in frequency domain (convolution theorem)
    result_fft = signal_fft * kernel_fft

    # Inverse FFT
    result_full = torch.fft.ifft(result_fft, dim=-1)

    # Handle real output for real inputs
    if not signal.is_complex() and not kernel.is_complex():
        result_full = result_full.real

    # Extract same-size output (centered)
    # For same-mode: output[i] is centered at signal[i]
    start = (kernel_len - 1) // 2
    result = result_full[..., start : start + signal_len]

    return result


def continuous_wavelet_transform(
    input: Tensor,
    scales: Tensor,
    wavelet: str | Callable[[Tensor], Tensor] = "morlet",
    *,
    dim: int = -1,
    sampling_period: float = 1.0,
) -> Tensor:
    r"""Compute the continuous wavelet transform (CWT) of a signal.

    The CWT convolves the input signal with scaled and translated versions
    of a wavelet function to provide time-frequency (or time-scale) analysis.

    The CWT at scale :math:`s` and position :math:`\tau` is defined as:

    .. math::
        W_x(s, \tau) = \frac{1}{\sqrt{s}} \int_{-\infty}^{\infty}
        x(t) \psi^*\left(\frac{t - \tau}{s}\right) dt

    where :math:`\psi` is the mother wavelet and :math:`\psi^*` is its
    complex conjugate.

    Parameters
    ----------
    input : Tensor
        Input signal tensor of any shape. The transform is computed along ``dim``.
    scales : Tensor
        1-D tensor of positive scale values. Smaller scales correspond to
        higher frequencies, larger scales to lower frequencies.
        The relationship between scale and frequency depends on the wavelet.
    wavelet : str or Callable, optional
        The wavelet to use. Can be:

        - ``"morlet"``: Complex Morlet wavelet (default). Good for oscillatory
          signals. Provides good frequency resolution.
        - ``"mexican_hat"``: Mexican hat (Ricker) wavelet. Real-valued, good
          for detecting sharp changes.
        - ``"ricker"``: Alias for ``"mexican_hat"``.
        - A callable ``f(t) -> Tensor`` that defines a custom wavelet.

        Default: ``"morlet"``.
    dim : int, optional
        The dimension along which to compute the transform.
        Default: ``-1`` (last dimension).
    sampling_period : float, optional
        Sampling period of the signal (time between samples).
        Affects the time axis of the wavelet.
        Default: ``1.0``.

    Returns
    -------
    Tensor
        CWT coefficients. For input of shape ``(..., N, ...)``, the output
        has shape ``(..., num_scales, N)`` where the scale dimension is
        inserted before the signal dimension (which moves to last).
        For 1-D input of shape ``(N,)``, output is ``(num_scales, N)``.

    Raises
    ------
    ValueError
        If ``scales`` is not a 1-D tensor.
        If any scale value is not positive.
        If ``wavelet`` string is not recognized.

    Examples
    --------
    Basic CWT with Morlet wavelet:

    >>> x = torch.randn(256)
    >>> scales = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    >>> cwt = continuous_wavelet_transform(x, scales)
    >>> cwt.shape
    torch.Size([5, 256])

    CWT with Mexican hat wavelet:

    >>> cwt = continuous_wavelet_transform(x, scales, wavelet="mexican_hat")
    >>> cwt.shape
    torch.Size([5, 256])
    >>> cwt.is_complex()
    False

    Batched input:

    >>> x = torch.randn(4, 256)
    >>> cwt = continuous_wavelet_transform(x, scales, dim=-1)
    >>> cwt.shape
    torch.Size([4, 5, 256])

    Custom wavelet function:

    >>> def gaussian_wavelet(t):
    ...     return torch.exp(-t**2 / 2)
    >>> cwt = continuous_wavelet_transform(x, scales, wavelet=gaussian_wavelet)

    Notes
    -----
    **Wavelet Choice:**

    - **Morlet** wavelet is complex-valued and provides excellent frequency
      resolution. Use for analyzing oscillatory signals, detecting periodicities.

    - **Mexican hat** wavelet is real-valued and provides good time localization.
      Use for detecting edges, discontinuities, and sharp features.

    **Scale-Frequency Relationship:**

    For the Morlet wavelet with center frequency :math:`\omega_0 = 5`:

    .. math::
        f = \frac{\omega_0}{2\pi s \cdot dt}

    where :math:`s` is the scale and :math:`dt` is the sampling period.

    **Normalization:**

    The :math:`1/\sqrt{s}` factor ensures energy preservation across scales,
    making the CWT coefficients comparable at different scales.

    **Implementation:**

    The convolution is computed using FFT for efficiency, with complexity
    :math:`O(N \cdot M \cdot \log N)` where :math:`N` is the signal length
    and :math:`M` is the number of scales.

    See Also
    --------
    discrete_wavelet_transform : Discrete wavelet transform.
    short_time_fourier_transform : Short-time Fourier transform.

    References
    ----------
    .. [1] Mallat, S. (2009). A Wavelet Tour of Signal Processing.
           Academic Press.
    .. [2] Torrence, C. & Compo, G.P. (1998). A Practical Guide to
           Wavelet Analysis. Bulletin of the American Meteorological Society.
    """
    # Validate scales
    if scales.ndim != 1:
        raise ValueError(
            f"scales must be a 1-D tensor, got {scales.ndim}-D tensor"
        )

    if (scales <= 0).any():
        raise ValueError("All scale values must be positive")

    # Get wavelet function
    wavelet_fn = _get_wavelet_function(wavelet)

    # Normalize dimension
    ndim = input.ndim
    normalized_dim = dim if dim >= 0 else dim + ndim

    # Move transform dimension to last position for easier handling
    if normalized_dim != ndim - 1:
        x = input.movedim(normalized_dim, -1)
    else:
        x = input

    signal_len = x.shape[-1]
    num_scales = scales.shape[0]

    # Batch shape (everything except the signal dimension)
    batch_shape = x.shape[:-1]

    # Flatten batch dimensions
    if x.ndim > 1:
        x_flat = x.reshape(-1, signal_len)  # (batch, signal_len)
    else:
        x_flat = x.unsqueeze(0)  # (1, signal_len)

    batch_size = x_flat.shape[0]

    # Determine output dtype based on wavelet type
    # Test with a sample to see if wavelet is complex
    test_t = torch.tensor([0.0], dtype=x.dtype, device=x.device)
    test_wavelet = wavelet_fn(test_t)
    is_complex_wavelet = test_wavelet.is_complex()

    if is_complex_wavelet:
        if x.dtype == torch.float32:
            output_dtype = torch.complex64
        elif x.dtype == torch.float64:
            output_dtype = torch.complex128
        else:
            output_dtype = torch.complex64
    else:
        output_dtype = x.dtype

    # Allocate output tensor
    # Shape: (batch_size, num_scales, signal_len)
    result = torch.empty(
        (batch_size, num_scales, signal_len),
        dtype=output_dtype,
        device=x.device,
    )

    # Time array for wavelet evaluation
    # We need enough samples to capture the wavelet at each scale
    # The wavelet is evaluated at t/s where t is the time and s is the scale

    # For each scale, compute the CWT coefficients
    for i, scale in enumerate(scales):
        scale_val = scale.item()

        # Determine wavelet support
        # For Morlet and Mexican hat, effective support is roughly [-4, 4]
        # After scaling by s, support becomes [-4*s, 4*s]
        # Number of samples needed: 2 * 4 * s / dt + 1
        half_support = int(math.ceil(4.0 * scale_val / sampling_period))
        # Ensure minimum wavelet length
        half_support = max(half_support, 4)

        # Create time array for wavelet
        wavelet_len = 2 * half_support + 1
        t_wavelet = torch.linspace(
            -half_support * sampling_period,
            half_support * sampling_period,
            wavelet_len,
            dtype=x.dtype,
            device=x.device,
        )

        # Evaluate scaled wavelet: psi(t/s) / sqrt(s)
        t_scaled = t_wavelet / scale_val
        wavelet_vals = wavelet_fn(t_scaled) / math.sqrt(scale_val)

        # Conjugate the wavelet for the convolution (CWT definition uses psi*)
        if is_complex_wavelet:
            wavelet_vals = wavelet_vals.conj()

        # Convolve each signal in the batch with the wavelet
        for b in range(batch_size):
            result[b, i, :] = _fft_convolve(x_flat[b], wavelet_vals)

    # Reshape result to match input batch shape
    # Output shape: (batch..., num_scales, signal_len)
    if len(batch_shape) > 0:
        result = result.reshape(*batch_shape, num_scales, signal_len)
    else:
        result = result.squeeze(0)  # Remove batch dimension for 1-D input

    return result

"""Integral transforms for signal and data processing.

This module provides implementations of various integral transforms
used in signal processing, mathematical physics, and engineering.

Transforms
----------
fourier_transform, inverse_fourier_transform
    Discrete Fourier transform with padding and windowing support.
cosine_transform, inverse_cosine_transform
    Discrete Cosine Transform (DCT) types I-IV with padding and windowing.
fourier_cosine_transform, inverse_fourier_cosine_transform
    Aliases for cosine_transform/inverse_cosine_transform (backward compatible).
sine_transform, inverse_sine_transform
    Discrete Sine Transform (DST) types I-IV with padding and windowing.
fourier_sine_transform, inverse_fourier_sine_transform
    Aliases for sine_transform/inverse_sine_transform (backward compatible).
hilbert_transform, inverse_hilbert_transform
    Hilbert transform for analytic signal computation.
hartley_transform
    Hartley transform (real-to-real, self-inverse).
laplace_transform, inverse_laplace_transform
    Numerical Laplace transform using quadrature.
two_sided_laplace_transform, inverse_two_sided_laplace_transform
    Two-sided (bilateral) Laplace transform.
mellin_transform, inverse_mellin_transform
    Mellin transform for scale-invariant analysis.
hankel_transform, inverse_hankel_transform
    Hankel transform using Ogata's quasi-discrete algorithm.
radon_transform, inverse_radon_transform
    Radon transform for CT reconstruction.
short_time_fourier_transform, inverse_short_time_fourier_transform
    Short-time Fourier transform for time-frequency analysis.
gabor_transform, inverse_gabor_transform
    Gabor transform (STFT with Gaussian window) for optimal time-frequency localization.
discrete_wavelet_transform, inverse_discrete_wavelet_transform
    Discrete wavelet transform using filter bank convolution.
continuous_wavelet_transform, inverse_continuous_wavelet_transform
    Continuous wavelet transform with Morlet and Mexican hat wavelets.
convolution
    FFT-based convolution with various modes.
"""

from ._abel_transform import abel_transform
from ._continuous_wavelet_transform import continuous_wavelet_transform
from ._convolution import convolution
from ._cosine_transform import cosine_transform, fourier_cosine_transform
from ._discrete_wavelet_transform import discrete_wavelet_transform
from ._fourier_transform import fourier_transform
from ._gabor_transform import gabor_transform
from ._hankel_transform import hankel_transform
from ._hartley_transform import hartley_transform
from ._hilbert_transform import hilbert_transform
from ._inverse_abel_transform import inverse_abel_transform
from ._inverse_continuous_wavelet_transform import (
    inverse_continuous_wavelet_transform,
)
from ._inverse_cosine_transform import (
    inverse_cosine_transform,
    inverse_fourier_cosine_transform,
)
from ._inverse_discrete_wavelet_transform import (
    inverse_discrete_wavelet_transform,
)
from ._inverse_fourier_transform import inverse_fourier_transform
from ._inverse_gabor_transform import inverse_gabor_transform
from ._inverse_hankel_transform import inverse_hankel_transform
from ._inverse_hilbert_transform import inverse_hilbert_transform
from ._inverse_laplace_transform import inverse_laplace_transform
from ._inverse_mellin_transform import inverse_mellin_transform
from ._inverse_radon_transform import inverse_radon_transform
from ._inverse_short_time_fourier_transform import (
    inverse_short_time_fourier_transform,
)
from ._inverse_sine_transform import (
    inverse_fourier_sine_transform,
    inverse_sine_transform,
)
from ._inverse_two_sided_laplace_transform import (
    inverse_two_sided_laplace_transform,
)
from ._inverse_z_transform import inverse_z_transform
from ._laplace_transform import laplace_transform
from ._mellin_transform import mellin_transform
from ._radon_transform import radon_transform
from ._short_time_fourier_transform import short_time_fourier_transform
from ._sine_transform import fourier_sine_transform, sine_transform
from ._two_sided_laplace_transform import two_sided_laplace_transform
from ._z_transform import z_transform

__all__ = [
    # Fourier
    "fourier_transform",
    "inverse_fourier_transform",
    # Gabor
    "gabor_transform",
    "inverse_gabor_transform",
    # DCT (new names)
    "cosine_transform",
    "inverse_cosine_transform",
    # DCT (backward-compatible aliases)
    "fourier_cosine_transform",
    "inverse_fourier_cosine_transform",
    # DST (new names)
    "sine_transform",
    "inverse_sine_transform",
    # DST (backward-compatible aliases)
    "fourier_sine_transform",
    "inverse_fourier_sine_transform",
    # Hilbert
    "hilbert_transform",
    "inverse_hilbert_transform",
    # Hartley
    "hartley_transform",
    # Laplace
    "laplace_transform",
    "inverse_laplace_transform",
    # Two-sided Laplace
    "two_sided_laplace_transform",
    "inverse_two_sided_laplace_transform",
    # Mellin
    "mellin_transform",
    "inverse_mellin_transform",
    # Hankel
    "hankel_transform",
    "inverse_hankel_transform",
    # Abel
    "abel_transform",
    "inverse_abel_transform",
    # Z-transform
    "z_transform",
    "inverse_z_transform",
    # Radon
    "radon_transform",
    "inverse_radon_transform",
    # STFT
    "short_time_fourier_transform",
    "inverse_short_time_fourier_transform",
    # DWT
    "discrete_wavelet_transform",
    "inverse_discrete_wavelet_transform",
    # CWT
    "continuous_wavelet_transform",
    "inverse_continuous_wavelet_transform",
    # Convolution
    "convolution",
]

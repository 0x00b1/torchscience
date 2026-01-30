// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/conj.h>
#include <ATen/ops/fft_rfft.h>
#include <ATen/ops/fft_rfftfreq.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/real.h>
#include <ATen/ops/sum.h>
#include <torch/extension.h>

namespace torchscience::composite::spectral_estimation {

// ============================================================================
// periodogram
// ============================================================================
//
// Computes the power spectral density estimate using a single FFT of the
// entire (optionally windowed) signal.
//
//   P(f) = (1 / (fs * S)) * |FFT(x * w)|^2
//
// where S = sum(w^2) for 'density' scaling, or S = (sum(w))^2 for 'spectrum'.
//
// Parameters:
//   x       - Input signal tensor (..., N)
//   window  - Window tensor (N,). Pass ones(N) for no windowing.
//   fs      - Sampling frequency (scalar double)
//   scaling - 0 = 'density' (V^2/Hz), 1 = 'spectrum' (V^2)
//
// Returns:
//   Tuple of (freqs, Pxx):
//     freqs - Frequency bins (N/2+1,)
//     Pxx   - Power spectral density (..., N/2+1)

inline std::tuple<at::Tensor, at::Tensor> periodogram(
    const at::Tensor& x,
    const at::Tensor& window,
    double fs,
    int64_t scaling
) {
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()),
        "periodogram requires real floating-point input, got ", x.scalar_type()
    );
    TORCH_CHECK(
        x.dim() >= 1,
        "periodogram requires at least 1 dimension, got ", x.dim()
    );

    int64_t n = x.size(-1);

    TORCH_CHECK(
        window.dim() == 1 && window.size(0) == n,
        "periodogram window must be 1-D with length matching signal length (",
        n, "), got shape ", window.sizes()
    );

    TORCH_CHECK(
        scaling == 0 || scaling == 1,
        "periodogram scaling must be 0 ('density') or 1 ('spectrum'), got ",
        scaling
    );

    // Apply window
    at::Tensor xw = x * window;

    // Compute one-sided FFT
    at::Tensor X = at::fft_rfft(xw, /*n=*/c10::nullopt, /*dim=*/-1);

    // Power spectrum: |X|^2
    at::Tensor Pxx = at::real(X * at::conj(X));

    // Normalization
    at::Tensor scale;
    if (scaling == 0) {
        // density: divide by fs * sum(w^2)
        scale = fs * at::sum(at::pow(window, 2));
    } else {
        // spectrum: divide by sum(w)^2
        scale = at::pow(at::sum(window), 2);
    }
    Pxx = Pxx / scale;

    // Double the non-DC, non-Nyquist bins (one-sided spectrum)
    int64_t n_freqs = Pxx.size(-1);
    if (n % 2 == 0) {
        // Even: DC at 0, Nyquist at n_freqs-1
        auto inner = Pxx.narrow(-1, 1, n_freqs - 2);
        Pxx = at::cat({
            Pxx.narrow(-1, 0, 1),
            inner * 2.0,
            Pxx.narrow(-1, n_freqs - 1, 1)
        }, -1);
    } else {
        // Odd: DC at 0, no Nyquist
        auto inner = Pxx.narrow(-1, 1, n_freqs - 1);
        Pxx = at::cat({
            Pxx.narrow(-1, 0, 1),
            inner * 2.0
        }, -1);
    }

    // Frequency bins
    at::Tensor freqs = at::fft_rfftfreq(
        n, /*d=*/1.0 / fs,
        x.options().dtype(at::kDouble)
    );

    return std::make_tuple(freqs, Pxx);
}

// ============================================================================
// welch
// ============================================================================
//
// Welch's method: average modified periodograms of overlapping segments.
//
//   P_welch(f) = (1/K) * sum_k periodogram(x_k * w)
//
// Parameters:
//   x        - Input signal (..., N)
//   window   - Window tensor (nperseg,)
//   nperseg  - Segment length (int, equals window.size(0))
//   noverlap - Overlap between segments (int)
//   fs       - Sampling frequency (double)
//   scaling  - 0='density', 1='spectrum'
//
// Returns: (freqs, Pxx)

inline std::tuple<at::Tensor, at::Tensor> welch(
    const at::Tensor& x,
    const at::Tensor& window,
    int64_t nperseg,
    int64_t noverlap,
    double fs,
    int64_t scaling
) {
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()),
        "welch requires real floating-point input, got ", x.scalar_type()
    );
    TORCH_CHECK(
        x.dim() >= 1,
        "welch requires at least 1 dimension, got ", x.dim()
    );
    TORCH_CHECK(
        window.dim() == 1 && window.size(0) == nperseg,
        "welch window must be 1-D with length nperseg (", nperseg,
        "), got shape ", window.sizes()
    );
    TORCH_CHECK(
        noverlap >= 0 && noverlap < nperseg,
        "welch noverlap must be in [0, nperseg), got ", noverlap
    );
    TORCH_CHECK(
        scaling == 0 || scaling == 1,
        "welch scaling must be 0 ('density') or 1 ('spectrum'), got ", scaling
    );

    int64_t n = x.size(-1);
    int64_t step = nperseg - noverlap;
    int64_t n_segments = (n - nperseg) / step + 1;

    TORCH_CHECK(
        n_segments >= 1,
        "welch: signal length (", n, ") too short for nperseg=", nperseg,
        " with noverlap=", noverlap
    );

    // Extract overlapping segments: shape (..., n_segments, nperseg)
    at::Tensor segments = x.unfold(-1, nperseg, step);

    // Apply window: broadcast (nperseg,) over (..., n_segments, nperseg)
    at::Tensor windowed = segments * window;

    // FFT each segment
    at::Tensor X = at::fft_rfft(windowed, /*n=*/c10::nullopt, /*dim=*/-1);

    // Power: |X|^2
    at::Tensor Pxx = at::real(X * at::conj(X));

    // Normalization
    at::Tensor scale;
    if (scaling == 0) {
        scale = fs * at::sum(at::pow(window, 2));
    } else {
        scale = at::pow(at::sum(window), 2);
    }
    Pxx = Pxx / scale;

    // Double non-DC, non-Nyquist bins (one-sided)
    int64_t n_freqs = Pxx.size(-1);
    if (nperseg % 2 == 0) {
        auto inner = Pxx.narrow(-1, 1, n_freqs - 2);
        Pxx = at::cat({
            Pxx.narrow(-1, 0, 1),
            inner * 2.0,
            Pxx.narrow(-1, n_freqs - 1, 1)
        }, -1);
    } else {
        auto inner = Pxx.narrow(-1, 1, n_freqs - 1);
        Pxx = at::cat({
            Pxx.narrow(-1, 0, 1),
            inner * 2.0
        }, -1);
    }

    // Average over segments (dim=-2 is the segment dimension)
    Pxx = at::mean(Pxx, -2);

    // Frequency bins
    at::Tensor freqs = at::fft_rfftfreq(
        nperseg, /*d=*/1.0 / fs,
        x.options().dtype(at::kDouble)
    );

    return std::make_tuple(freqs, Pxx);
}

// ============================================================================
// spectrogram
// ============================================================================
//
// Computes the spectrogram: a time-frequency representation using
// short-time Fourier transform magnitudes.
//
// Returns (freqs, times, Sxx):
//   freqs - (nperseg/2+1,)
//   times - (n_segments,)
//   Sxx   - (..., nperseg/2+1, n_segments)

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> spectrogram(
    const at::Tensor& x,
    const at::Tensor& window,
    int64_t nperseg,
    int64_t noverlap,
    double fs,
    int64_t scaling
) {
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()),
        "spectrogram requires real floating-point input, got ", x.scalar_type()
    );
    TORCH_CHECK(
        x.dim() >= 1,
        "spectrogram requires at least 1 dimension, got ", x.dim()
    );
    TORCH_CHECK(
        window.dim() == 1 && window.size(0) == nperseg,
        "spectrogram window must be 1-D with length nperseg (", nperseg,
        "), got shape ", window.sizes()
    );
    TORCH_CHECK(
        noverlap >= 0 && noverlap < nperseg,
        "spectrogram noverlap must be in [0, nperseg), got ", noverlap
    );
    TORCH_CHECK(
        scaling == 0 || scaling == 1,
        "spectrogram scaling must be 0 ('density') or 1 ('spectrum'), got ",
        scaling
    );

    int64_t n = x.size(-1);
    int64_t step = nperseg - noverlap;
    int64_t n_segments = (n - nperseg) / step + 1;

    TORCH_CHECK(
        n_segments >= 1,
        "spectrogram: signal length (", n, ") too short for nperseg=", nperseg,
        " with noverlap=", noverlap
    );

    // Extract segments: (..., n_segments, nperseg)
    at::Tensor segments = x.unfold(-1, nperseg, step);

    // Apply window
    at::Tensor windowed = segments * window;

    // FFT each segment
    at::Tensor X = at::fft_rfft(windowed, /*n=*/c10::nullopt, /*dim=*/-1);

    // Power spectrum
    at::Tensor Sxx = at::real(X * at::conj(X));

    // Normalization
    at::Tensor scale;
    if (scaling == 0) {
        scale = fs * at::sum(at::pow(window, 2));
    } else {
        scale = at::pow(at::sum(window), 2);
    }
    Sxx = Sxx / scale;

    // Double non-DC, non-Nyquist bins
    int64_t n_freqs = Sxx.size(-1);
    if (nperseg % 2 == 0) {
        auto inner = Sxx.narrow(-1, 1, n_freqs - 2);
        Sxx = at::cat({
            Sxx.narrow(-1, 0, 1),
            inner * 2.0,
            Sxx.narrow(-1, n_freqs - 1, 1)
        }, -1);
    } else {
        auto inner = Sxx.narrow(-1, 1, n_freqs - 1);
        Sxx = at::cat({
            Sxx.narrow(-1, 0, 1),
            inner * 2.0
        }, -1);
    }

    // Transpose last two dims to get (..., n_freqs, n_segments)
    // Current shape is (..., n_segments, n_freqs)
    int64_t ndim = Sxx.dim();
    Sxx = Sxx.transpose(ndim - 2, ndim - 1);

    // Frequency bins
    at::Tensor freqs = at::fft_rfftfreq(
        nperseg, /*d=*/1.0 / fs,
        x.options().dtype(at::kDouble)
    );

    // Time bins: center of each segment
    at::Tensor times = at::arange(n_segments, x.options().dtype(at::kDouble))
        * static_cast<double>(step) / fs
        + static_cast<double>(nperseg) / (2.0 * fs);

    return std::make_tuple(freqs, times, Sxx);
}

// ============================================================================
// cross_spectral_density
// ============================================================================
//
// Cross spectral density using Welch's method.
//
//   P_xy(f) = (1/K) * sum_k conj(FFT(x_k * w)) * FFT(y_k * w) / (fs * S)
//
// Parameters:
//   x, y     - Input signals (..., N) - must be broadcastable
//   window   - Window tensor (nperseg,)
//   nperseg  - Segment length
//   noverlap - Overlap
//   fs       - Sampling frequency
//   scaling  - 0='density', 1='spectrum'
//
// Returns: (freqs, Pxy) where Pxy is complex

inline std::tuple<at::Tensor, at::Tensor> cross_spectral_density(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& window,
    int64_t nperseg,
    int64_t noverlap,
    double fs,
    int64_t scaling
) {
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()),
        "cross_spectral_density requires real floating-point input for x, got ",
        x.scalar_type()
    );
    TORCH_CHECK(
        at::isFloatingType(y.scalar_type()),
        "cross_spectral_density requires real floating-point input for y, got ",
        y.scalar_type()
    );
    TORCH_CHECK(
        x.dim() >= 1 && y.dim() >= 1,
        "cross_spectral_density requires at least 1 dimension"
    );
    TORCH_CHECK(
        window.dim() == 1 && window.size(0) == nperseg,
        "cross_spectral_density window must be 1-D with length nperseg (",
        nperseg, "), got shape ", window.sizes()
    );
    TORCH_CHECK(
        noverlap >= 0 && noverlap < nperseg,
        "cross_spectral_density noverlap must be in [0, nperseg), got ",
        noverlap
    );
    TORCH_CHECK(
        scaling == 0 || scaling == 1,
        "cross_spectral_density scaling must be 0 or 1, got ", scaling
    );

    int64_t step = nperseg - noverlap;

    // Segment both signals
    at::Tensor x_seg = x.unfold(-1, nperseg, step);
    at::Tensor y_seg = y.unfold(-1, nperseg, step);

    // Apply window
    at::Tensor xw = x_seg * window;
    at::Tensor yw = y_seg * window;

    // FFT each segment
    at::Tensor Xf = at::fft_rfft(xw, /*n=*/c10::nullopt, /*dim=*/-1);
    at::Tensor Yf = at::fft_rfft(yw, /*n=*/c10::nullopt, /*dim=*/-1);

    // Cross spectrum: conj(X) * Y
    at::Tensor Pxy = at::conj(Xf) * Yf;

    // Normalization
    at::Tensor scale;
    if (scaling == 0) {
        scale = fs * at::sum(at::pow(window, 2));
    } else {
        scale = at::pow(at::sum(window), 2);
    }
    Pxy = Pxy / scale;

    // Double non-DC, non-Nyquist bins (one-sided)
    int64_t n_freqs = Pxy.size(-1);
    if (nperseg % 2 == 0) {
        auto inner = Pxy.narrow(-1, 1, n_freqs - 2);
        Pxy = at::cat({
            Pxy.narrow(-1, 0, 1),
            inner * 2.0,
            Pxy.narrow(-1, n_freqs - 1, 1)
        }, -1);
    } else {
        auto inner = Pxy.narrow(-1, 1, n_freqs - 1);
        Pxy = at::cat({
            Pxy.narrow(-1, 0, 1),
            inner * 2.0
        }, -1);
    }

    // Average over segments
    Pxy = at::mean(Pxy, -2);

    // Frequency bins
    at::Tensor freqs = at::fft_rfftfreq(
        nperseg, /*d=*/1.0 / fs,
        x.options().dtype(at::kDouble)
    );

    return std::make_tuple(freqs, Pxy);
}

}  // namespace torchscience::composite::spectral_estimation

// =============================================================================
// CompositeImplicitAutograd registrations
// Gradients flow automatically through ATen FFT operations
// =============================================================================

TORCH_LIBRARY_IMPL(torchscience, CompositeImplicitAutograd, m) {
    m.impl(
        "periodogram",
        &torchscience::composite::spectral_estimation::periodogram
    );
    m.impl(
        "welch",
        &torchscience::composite::spectral_estimation::welch
    );
    m.impl(
        "spectrogram_psd",
        &torchscience::composite::spectral_estimation::spectrogram
    );
    m.impl(
        "cross_spectral_density",
        &torchscience::composite::spectral_estimation::cross_spectral_density
    );
}

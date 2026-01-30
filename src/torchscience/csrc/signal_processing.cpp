// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// Window functions
#include "composite/signal_processing/window_functions.h"

// Spectral estimation
#include "composite/signal_processing/spectral_estimation.h"
#include "cpu/signal_processing/window_functions.h"
#include "meta/signal_processing/window_functions.h"
#include "autograd/signal_processing/window_functions.h"

// Waveforms
#include "cpu/signal_processing/waveform/sine_wave.h"
#include "cpu/signal_processing/waveform/sine_wave_backward.h"
#include "cpu/signal_processing/waveform/square_wave.h"
#include "cpu/signal_processing/waveform/sawtooth_wave.h"
#include "cpu/signal_processing/waveform/triangle_wave.h"
#include "cpu/signal_processing/waveform/pulse_wave.h"
#include "cpu/signal_processing/waveform/impulse_wave.h"
#include "cpu/signal_processing/waveform/step_wave.h"
#include "cpu/signal_processing/waveform/ramp_wave.h"
#include "cpu/signal_processing/waveform/gaussian_pulse_wave.h"
#include "cpu/signal_processing/waveform/sinc_pulse_wave.h"
#include "cpu/signal_processing/waveform/linear_chirp_wave.h"
#include "cpu/signal_processing/waveform/logarithmic_chirp_wave.h"
#include "cpu/signal_processing/waveform/hyperbolic_chirp_wave.h"
#include "cpu/signal_processing/waveform/frequency_modulated_wave.h"
#include "meta/signal_processing/waveform/sine_wave.h"
#include "autograd/signal_processing/waveform/sine_wave.h"

// Noise
#include "cpu/signal_processing/noise/white_noise.h"
#include "cpu/signal_processing/noise/pink_noise.h"
#include "cpu/signal_processing/noise/brown_noise.h"
#include "cpu/signal_processing/noise/blue_noise.h"
#include "cpu/signal_processing/noise/violet_noise.h"
#include "cpu/signal_processing/noise/poisson_noise.h"
#include "cpu/signal_processing/noise/shot_noise.h"
#include "cpu/signal_processing/noise/impulse_noise.h"

// Filter
#include "cpu/signal_processing/filter.h"
#include "meta/signal_processing/filter.h"
#include "autograd/signal_processing/filter.h"
#include "autocast/signal_processing/filter.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Filter
  m.def("butterworth_analog_bandpass_filter(int n, Tensor omega_p1, Tensor omega_p2) -> Tensor");
  m.def("butterworth_analog_bandpass_filter_backward(Tensor grad_output, int n, Tensor omega_p1, Tensor omega_p2) -> (Tensor, Tensor)");
  m.def("butterworth_analog_bandpass_filter_backward_backward(Tensor gg_omega_p1, Tensor gg_omega_p2, Tensor grad_output, int n, Tensor omega_p1, Tensor omega_p2) -> (Tensor, Tensor, Tensor)");

  // Waveforms
  m.def("sine_wave(int? n=None, Tensor? t=None, *, "
        "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");
  m.def("sine_wave_backward(Tensor grad_output, int? n, Tensor? t, "
        "Tensor frequency, float sample_rate, Tensor amplitude, Tensor phase) -> "
        "(Tensor, Tensor, Tensor, Tensor)");

  m.def("square_wave(int? n=None, Tensor? t=None, *, "
        "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, Tensor duty, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("sawtooth_wave(int? n=None, Tensor? t=None, *, "
        "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("triangle_wave(int? n=None, Tensor? t=None, *, "
        "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("pulse_wave(int? n=None, Tensor? t=None, *, "
        "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, Tensor duty_cycle, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("impulse_wave(int n, *, "
        "Tensor position, Tensor amplitude, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("step_wave(int n, *, "
        "Tensor position, Tensor amplitude, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("ramp_wave(int n, *, "
        "Tensor position, Tensor slope, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("gaussian_pulse_wave(int n, *, "
        "Tensor center, Tensor std, Tensor amplitude, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("sinc_pulse_wave(int n, *, "
        "Tensor center, Tensor bandwidth, Tensor amplitude, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("linear_chirp_wave(int? n=None, Tensor? t=None, *, "
        "Tensor f0, Tensor f1, float t1=1.0, float sample_rate=1.0, "
        "Tensor amplitude, Tensor phase, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("logarithmic_chirp_wave(int? n=None, Tensor? t=None, *, "
        "Tensor f0, Tensor f1, float t1=1.0, float sample_rate=1.0, "
        "Tensor amplitude, Tensor phase, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("hyperbolic_chirp_wave(int? n=None, Tensor? t=None, *, "
        "Tensor f0, Tensor f1, float t1=1.0, float sample_rate=1.0, "
        "Tensor amplitude, Tensor phase, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  // Frequency modulated waves
  m.def("frequency_modulated_wave(int? n=None, Tensor? t=None, *, "
        "Tensor carrier_frequency, Tensor modulator_frequency, Tensor modulation_index, "
        "float sample_rate=1.0, Tensor amplitude, Tensor phase, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  m.def("frequency_modulated_wave_arbitrary(int? n=None, Tensor? t=None, *, "
        "Tensor carrier_frequency, Tensor modulating_signal, Tensor modulation_index, "
        "float sample_rate=1.0, Tensor amplitude, Tensor phase, "
        "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  // Window functions - parameterless
  m.def("rectangular_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("hann_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_hann_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("hamming_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_hamming_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("blackman_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_blackman_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("bartlett_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_bartlett_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("cosine_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_cosine_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("nuttall_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_nuttall_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("triangular_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_triangular_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("welch_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_welch_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("parzen_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_parzen_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("blackman_harris_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_blackman_harris_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("flat_top_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_flat_top_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("sine_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_sine_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("bartlett_hann_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_bartlett_hann_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("lanczos_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  m.def("periodic_lanczos_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");

  // Window functions - parameterized (Gaussian)
  m.def("gaussian_window(int n, Tensor std, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_gaussian_window(int n, Tensor std, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("gaussian_window_backward(Tensor grad_output, Tensor output, int n, Tensor std) -> Tensor");
  m.def("periodic_gaussian_window_backward(Tensor grad_output, Tensor output, int n, Tensor std) -> Tensor");

  // Window functions - parameterized (General Hamming)
  m.def("general_hamming_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_general_hamming_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("general_hamming_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");
  m.def("periodic_general_hamming_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");

  // Window functions - parameterized (General Cosine)
  m.def("general_cosine_window(int n, Tensor coeffs, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_general_cosine_window(int n, Tensor coeffs, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("general_cosine_window_backward(Tensor grad_output, Tensor output, int n, Tensor coeffs) -> Tensor");
  m.def("periodic_general_cosine_window_backward(Tensor grad_output, Tensor output, int n, Tensor coeffs) -> Tensor");

  // Window functions - parameterized (Tukey)
  m.def("tukey_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_tukey_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("tukey_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");
  m.def("periodic_tukey_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");

  // Window functions - parameterized (Exponential)
  m.def("exponential_window(int n, Tensor tau, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_exponential_window(int n, Tensor tau, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("exponential_window_backward(Tensor grad_output, Tensor output, int n, Tensor tau) -> Tensor");
  m.def("periodic_exponential_window_backward(Tensor grad_output, Tensor output, int n, Tensor tau) -> Tensor");

  // Window functions - parameterized (Hann-Poisson)
  m.def("hann_poisson_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_hann_poisson_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("hann_poisson_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");
  m.def("periodic_hann_poisson_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");

  // Window functions - parameterized (Generalized Normal)
  m.def("generalized_normal_window(int n, Tensor p, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_generalized_normal_window(int n, Tensor p, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("generalized_normal_window_backward(Tensor grad_output, Tensor output, int n, Tensor p, Tensor sigma) -> (Tensor, Tensor)");
  m.def("periodic_generalized_normal_window_backward(Tensor grad_output, Tensor output, int n, Tensor p, Tensor sigma) -> (Tensor, Tensor)");

  // Window functions - parameterized (Kaiser)
  m.def("kaiser_window(int n, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_kaiser_window(int n, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("kaiser_window_backward(Tensor grad_output, Tensor output, int n, Tensor beta) -> Tensor");
  m.def("periodic_kaiser_window_backward(Tensor grad_output, Tensor output, int n, Tensor beta) -> Tensor");

  // Window functions - parameterized (Planck-taper)
  m.def("planck_taper_window(int n, Tensor epsilon, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_planck_taper_window(int n, Tensor epsilon, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("planck_taper_window_backward(Tensor grad_output, Tensor output, int n, Tensor epsilon) -> Tensor");
  m.def("periodic_planck_taper_window_backward(Tensor grad_output, Tensor output, int n, Tensor epsilon) -> Tensor");

  // Window functions - parameterized (Planck-Bessel)
  m.def("planck_bessel_window(int n, Tensor epsilon, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_planck_bessel_window(int n, Tensor epsilon, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("planck_bessel_window_backward(Tensor grad_output, Tensor output, int n, Tensor epsilon, Tensor beta) -> (Tensor, Tensor)");
  m.def("periodic_planck_bessel_window_backward(Tensor grad_output, Tensor output, int n, Tensor epsilon, Tensor beta) -> (Tensor, Tensor)");

  // Window functions - FFT-based (Dolph-Chebyshev)
  m.def("dolph_chebyshev_window(int n, Tensor attenuation, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_dolph_chebyshev_window(int n, Tensor attenuation, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");

  // Window functions - FFT-based (Ultraspherical)
  m.def("ultraspherical_window(int n, Tensor mu, Tensor x_mu, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_ultraspherical_window(int n, Tensor mu, Tensor x_mu, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");

  // Window functions - eigenvalue-based and polynomial
  m.def("discrete_prolate_spheroidal_sequence_window(int n, Tensor nw, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_discrete_prolate_spheroidal_sequence_window(int n, Tensor nw, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("approximate_confined_gaussian_window(int n, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_approximate_confined_gaussian_window(int n, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("confined_gaussian_window(int n, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_confined_gaussian_window(int n, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("generalized_adaptive_polynomial_window(int n, Tensor alpha, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  m.def("periodic_generalized_adaptive_polynomial_window(int n, Tensor alpha, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");

  // Noise functions
  m.def("white_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  m.def("pink_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  m.def("brown_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  m.def("blue_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  m.def("violet_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  m.def("poisson_noise(int[] size, Tensor rate, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, Generator? generator=None) -> Tensor");
  m.def("shot_noise(int[] size, Tensor rate, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  m.def("impulse_noise(int[] size, Tensor p_salt, Tensor p_pepper, float salt_value, float pepper_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, Generator? generator=None) -> Tensor");

  // Spectral estimation
  m.def("periodogram(Tensor x, Tensor window, float fs, int scaling) -> (Tensor, Tensor)");
  m.def("welch(Tensor x, Tensor window, int nperseg, int noverlap, float fs, int scaling) -> (Tensor, Tensor)");
}

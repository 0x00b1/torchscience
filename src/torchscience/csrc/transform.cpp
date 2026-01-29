// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/transform/fourier_transform.h"
#include "cpu/transform/inverse_fourier_transform.h"
#include "cpu/transform/fourier_cosine_transform.h"
#include "cpu/transform/inverse_fourier_cosine_transform.h"
#include "cpu/transform/fourier_sine_transform.h"
#include "cpu/transform/inverse_fourier_sine_transform.h"
#include "cpu/transform/convolution.h"
#include "cpu/transform/laplace_transform.h"
#include "cpu/transform/inverse_laplace_transform.h"
#include "cpu/transform/mellin_transform.h"
#include "cpu/transform/inverse_mellin_transform.h"
#include "cpu/transform/two_sided_laplace_transform.h"
#include "cpu/transform/inverse_two_sided_laplace_transform.h"
#include "cpu/transform/hankel_transform.h"
#include "cpu/transform/inverse_hankel_transform.h"
#include "cpu/transform/abel_transform.h"
#include "cpu/transform/inverse_abel_transform.h"
#include "cpu/transform/z_transform.h"
#include "cpu/transform/inverse_z_transform.h"
#include "cpu/transform/radon_transform.h"
#include "cpu/transform/inverse_radon_transform.h"
#include "cpu/transform/discrete_wavelet_transform.h"
#include "cpu/transform/inverse_discrete_wavelet_transform.h"

// Meta backend
#include "meta/transform/fourier_transform.h"
#include "meta/transform/inverse_fourier_transform.h"
#include "meta/transform/fourier_cosine_transform.h"
#include "meta/transform/fourier_sine_transform.h"
#include "meta/transform/convolution.h"
#include "meta/transform/laplace_transform.h"
#include "meta/transform/inverse_laplace_transform.h"
#include "meta/transform/mellin_transform.h"
#include "meta/transform/inverse_mellin_transform.h"
#include "meta/transform/two_sided_laplace_transform.h"
#include "meta/transform/inverse_two_sided_laplace_transform.h"
#include "meta/transform/hankel_transform.h"
#include "meta/transform/inverse_hankel_transform.h"
#include "meta/transform/abel_transform.h"
#include "meta/transform/inverse_abel_transform.h"
#include "meta/transform/z_transform.h"
#include "meta/transform/inverse_z_transform.h"
#include "meta/transform/radon_transform.h"
#include "meta/transform/inverse_radon_transform.h"
#include "meta/transform/discrete_wavelet_transform.h"
#include "meta/transform/inverse_discrete_wavelet_transform.h"

// Autograd backend
#include "autograd/transform/fourier_transform.h"
#include "autograd/transform/inverse_fourier_transform.h"
#include "autograd/transform/fourier_cosine_transform.h"
#include "autograd/transform/fourier_sine_transform.h"
#include "autograd/transform/convolution.h"
#include "autograd/transform/laplace_transform.h"
#include "autograd/transform/inverse_laplace_transform.h"
#include "autograd/transform/mellin_transform.h"
#include "autograd/transform/inverse_mellin_transform.h"
#include "autograd/transform/two_sided_laplace_transform.h"
#include "autograd/transform/inverse_two_sided_laplace_transform.h"
#include "autograd/transform/hankel_transform.h"
#include "autograd/transform/inverse_hankel_transform.h"
#include "autograd/transform/abel_transform.h"
#include "autograd/transform/inverse_abel_transform.h"
#include "autograd/transform/z_transform.h"
#include "autograd/transform/inverse_z_transform.h"
#include "autograd/transform/radon_transform.h"
#include "autograd/transform/inverse_radon_transform.h"
#include "autograd/transform/discrete_wavelet_transform.h"
#include "autograd/transform/inverse_discrete_wavelet_transform.h"

// Autocast backend
#include "autocast/transform/fourier_transform.h"
#include "autocast/transform/inverse_fourier_transform.h"
#include "autocast/transform/fourier_cosine_transform.h"
#include "autocast/transform/inverse_fourier_cosine_transform.h"
#include "autocast/transform/fourier_sine_transform.h"
#include "autocast/transform/inverse_fourier_sine_transform.h"
#include "autocast/transform/convolution.h"
#include "autocast/transform/laplace_transform.h"
#include "autocast/transform/inverse_laplace_transform.h"
#include "autocast/transform/two_sided_laplace_transform.h"
#include "autocast/transform/inverse_two_sided_laplace_transform.h"
#include "autocast/transform/mellin_transform.h"
#include "autocast/transform/inverse_mellin_transform.h"
#include "autocast/transform/hankel_transform.h"
#include "autocast/transform/inverse_hankel_transform.h"
#include "autocast/transform/abel_transform.h"
#include "autocast/transform/inverse_abel_transform.h"
#include "autocast/transform/z_transform.h"
#include "autocast/transform/inverse_z_transform.h"
#include "autocast/transform/radon_transform.h"
#include "autocast/transform/inverse_radon_transform.h"
#include "autocast/transform/discrete_wavelet_transform.h"
#include "autocast/transform/inverse_discrete_wavelet_transform.h"

#ifdef TORCHSCIENCE_CUDA
#include "cuda/transform/fourier_transform.cu"
#include "cuda/transform/inverse_fourier_transform.cu"
#endif

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Fourier transform
  m.def("fourier_transform(Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window, int norm) -> Tensor");
  m.def("fourier_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window, int norm) -> Tensor");
  m.def("fourier_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window, int norm) -> (Tensor, Tensor)");

  m.def("inverse_fourier_transform(Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window, int norm) -> Tensor");
  m.def("inverse_fourier_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window, int norm) -> Tensor");
  m.def("inverse_fourier_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window, int norm) -> (Tensor, Tensor)");

  // Fourier cosine transform
  m.def("fourier_cosine_transform(Tensor input, int n_param, int dim, int type, int norm) -> Tensor");
  m.def("fourier_cosine_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int type, int norm) -> Tensor");
  m.def("fourier_cosine_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int type, int norm) -> (Tensor, Tensor)");

  m.def("inverse_fourier_cosine_transform(Tensor input, int n_param, int dim, int type, int norm) -> Tensor");
  m.def("inverse_fourier_cosine_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int type, int norm) -> Tensor");
  m.def("inverse_fourier_cosine_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int type, int norm) -> (Tensor, Tensor)");

  // Fourier sine transform
  m.def("fourier_sine_transform(Tensor input, int n_param, int dim, int type, int norm) -> Tensor");
  m.def("fourier_sine_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int type, int norm) -> Tensor");
  m.def("fourier_sine_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int type, int norm) -> (Tensor, Tensor)");

  m.def("inverse_fourier_sine_transform(Tensor input, int n_param, int dim, int type, int norm) -> Tensor");
  m.def("inverse_fourier_sine_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int type, int norm) -> Tensor");
  m.def("inverse_fourier_sine_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int type, int norm) -> (Tensor, Tensor)");

  // Convolution
  m.def("convolution(Tensor input, Tensor kernel, int dim, int mode) -> Tensor");
  m.def("convolution_backward(Tensor grad_output, Tensor input, Tensor kernel, int dim, int mode) -> (Tensor, Tensor)");
  m.def("convolution_backward_backward(Tensor gg_input, Tensor gg_kernel, Tensor grad_output, Tensor input, Tensor kernel, int dim, int mode) -> (Tensor, Tensor, Tensor)");

  // Laplace transform
  m.def("laplace_transform(Tensor input, Tensor s, Tensor t, int dim, int integration_method) -> Tensor");
  m.def("laplace_transform_backward(Tensor grad_output, Tensor input, Tensor s, Tensor t, int dim, int integration_method) -> Tensor");
  m.def("laplace_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor s, Tensor t, int dim, int integration_method) -> (Tensor, Tensor)");

  m.def("inverse_laplace_transform(Tensor input, Tensor t_out, Tensor s_in, int dim, float sigma, int integration_method) -> Tensor");
  m.def("inverse_laplace_transform_backward(Tensor grad_output, Tensor input, Tensor t_out, Tensor s_in, int dim, float sigma, int integration_method) -> Tensor");
  m.def("inverse_laplace_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor t_out, Tensor s_in, int dim, float sigma, int integration_method) -> (Tensor, Tensor)");

  // Two-sided Laplace transform
  m.def("two_sided_laplace_transform(Tensor input, Tensor s, Tensor t, int dim, int integration_method) -> Tensor");
  m.def("two_sided_laplace_transform_backward(Tensor grad_output, Tensor input, Tensor s, Tensor t, int dim, int integration_method) -> Tensor");
  m.def("two_sided_laplace_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor s, Tensor t, int dim, int integration_method) -> (Tensor, Tensor)");

  m.def("inverse_two_sided_laplace_transform(Tensor input, Tensor t_out, Tensor s_in, int dim, float sigma, int integration_method) -> Tensor");
  m.def("inverse_two_sided_laplace_transform_backward(Tensor grad_output, Tensor input, Tensor t_out, Tensor s_in, int dim, float sigma, int integration_method) -> Tensor");
  m.def("inverse_two_sided_laplace_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor t_out, Tensor s_in, int dim, float sigma, int integration_method) -> (Tensor, Tensor)");

  // Mellin transform
  m.def("mellin_transform(Tensor input, Tensor s, Tensor t, int dim, int integration_method) -> Tensor");
  m.def("mellin_transform_backward(Tensor grad_output, Tensor input, Tensor s, Tensor t, int dim, int integration_method) -> Tensor");
  m.def("mellin_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor s, Tensor t, int dim, int integration_method) -> (Tensor, Tensor)");

  m.def("inverse_mellin_transform(Tensor input, Tensor t_out, Tensor s_in, int dim, float c, int integration_method) -> Tensor");
  m.def("inverse_mellin_transform_backward(Tensor grad_output, Tensor input, Tensor t_out, Tensor s_in, int dim, float c, int integration_method) -> Tensor");
  m.def("inverse_mellin_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor t_out, Tensor s_in, int dim, float c, int integration_method) -> (Tensor, Tensor)");

  // Hankel transform
  m.def("hankel_transform(Tensor input, Tensor k_out, Tensor r_in, int dim, float order, int integration_method) -> Tensor");
  m.def("hankel_transform_backward(Tensor grad_output, Tensor input, Tensor k_out, Tensor r_in, int dim, float order, int integration_method) -> Tensor");
  m.def("hankel_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor k_out, Tensor r_in, int dim, float order, int integration_method) -> (Tensor, Tensor)");

  m.def("inverse_hankel_transform(Tensor input, Tensor r_out, Tensor k_in, int dim, float order, int integration_method) -> Tensor");
  m.def("inverse_hankel_transform_backward(Tensor grad_output, Tensor input, Tensor r_out, Tensor k_in, int dim, float order, int integration_method) -> Tensor");
  m.def("inverse_hankel_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor r_out, Tensor k_in, int dim, float order, int integration_method) -> (Tensor, Tensor)");

  // Abel transform
  m.def("abel_transform(Tensor input, Tensor y_out, Tensor r_in, int dim, int integration_method) -> Tensor");
  m.def("abel_transform_backward(Tensor grad_output, Tensor input, Tensor y_out, Tensor r_in, int dim, int integration_method) -> Tensor");
  m.def("abel_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor y_out, Tensor r_in, int dim, int integration_method) -> (Tensor, Tensor)");

  m.def("inverse_abel_transform(Tensor input, Tensor r_out, Tensor y_in, int dim, int integration_method) -> Tensor");
  m.def("inverse_abel_transform_backward(Tensor grad_output, Tensor input, Tensor r_out, Tensor y_in, int dim, int integration_method) -> Tensor");
  m.def("inverse_abel_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor r_out, Tensor y_in, int dim, int integration_method) -> (Tensor, Tensor)");

  // Z-transform
  m.def("z_transform(Tensor input, Tensor z_out, int dim) -> Tensor");
  m.def("z_transform_backward(Tensor grad_output, Tensor input, Tensor z_out, int dim) -> Tensor");
  m.def("z_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor z_out, int dim) -> (Tensor, Tensor)");

  m.def("inverse_z_transform(Tensor input, Tensor n_out, Tensor z_in, int dim) -> Tensor");
  m.def("inverse_z_transform_backward(Tensor grad_output, Tensor input, Tensor n_out, Tensor z_in, int dim) -> Tensor");
  m.def("inverse_z_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor n_out, Tensor z_in, int dim) -> (Tensor, Tensor)");

  // Radon transform
  m.def("radon_transform(Tensor input, Tensor angles, bool circle) -> Tensor");
  m.def("radon_transform_backward(Tensor grad_output, Tensor input, Tensor angles, bool circle) -> Tensor");
  m.def("radon_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor angles, bool circle) -> (Tensor, Tensor)");

  m.def("inverse_radon_transform(Tensor sinogram, Tensor angles, bool circle, int output_size, int filter_type) -> Tensor");
  m.def("inverse_radon_transform_backward(Tensor grad_output, Tensor sinogram, Tensor angles, bool circle, int output_size, int filter_type) -> Tensor");
  m.def("inverse_radon_transform_backward_backward(Tensor gg_sinogram, Tensor grad_output, Tensor sinogram, Tensor angles, bool circle, int output_size, int filter_type) -> (Tensor, Tensor)");

  // Discrete wavelet transform
  m.def("discrete_wavelet_transform(Tensor input, Tensor filter_lo, Tensor filter_hi, int levels, int mode) -> Tensor");
  m.def("discrete_wavelet_transform_backward(Tensor grad_output, Tensor input, Tensor filter_lo, Tensor filter_hi, int levels, int mode, int input_length) -> Tensor");
  m.def("discrete_wavelet_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, Tensor filter_lo, Tensor filter_hi, int levels, int mode, int input_length) -> (Tensor, Tensor)");

  m.def("inverse_discrete_wavelet_transform(Tensor coeffs, Tensor filter_lo, Tensor filter_hi, int levels, int mode, int output_length) -> Tensor");
  m.def("inverse_discrete_wavelet_transform_backward(Tensor grad_output, Tensor coeffs, Tensor filter_lo, Tensor filter_hi, int levels, int mode, int output_length) -> Tensor");
  m.def("inverse_discrete_wavelet_transform_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs, Tensor filter_lo, Tensor filter_hi, int levels, int mode, int output_length) -> (Tensor, Tensor)");
}

// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/special_functions.h"

// Meta backend
#include "meta/special_functions.h"

// Autograd backend
#include "autograd/special_functions.h"

// Autocast backend
#include "autocast/special_functions.h"

// Sparse backends
#include "sparse/coo/cpu/special_functions.h"
#include "sparse/csr/cpu/special_functions.h"

// Quantized backends
#include "quantized/cpu/special_functions.h"

#ifdef TORCHSCIENCE_CUDA
#include "sparse/coo/cuda/special_functions.h"
#include "sparse/csr/cuda/special_functions.h"
#include "quantized/cuda/special_functions.h"
#endif

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Gamma function and derivatives
  m.def("gamma(Tensor z) -> Tensor");
  m.def("gamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("gamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("digamma(Tensor z) -> Tensor");
  m.def("digamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("digamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("trigamma(Tensor z) -> Tensor");
  m.def("trigamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("trigamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("polygamma(Tensor n, Tensor z) -> Tensor");
  m.def("polygamma_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("polygamma_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  m.def("log_gamma(Tensor z) -> Tensor");
  m.def("log_gamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("log_gamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Beta function
  m.def("beta(Tensor a, Tensor b) -> Tensor");
  m.def("beta_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  m.def("beta_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  m.def("log_beta(Tensor a, Tensor b) -> Tensor");
  m.def("log_beta_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  m.def("log_beta_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  m.def("incomplete_beta(Tensor x, Tensor a, Tensor b) -> Tensor");
  m.def("incomplete_beta_backward(Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("incomplete_beta_backward_backward(Tensor gg_x, Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");

  // Regularized incomplete gamma functions
  m.def("regularized_gamma_p(Tensor a, Tensor x) -> Tensor");
  m.def("regularized_gamma_p_backward(Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor)");
  m.def("regularized_gamma_p_backward_backward(Tensor grad_grad_a, Tensor grad_grad_x, Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor, Tensor)");

  m.def("regularized_gamma_q(Tensor a, Tensor x) -> Tensor");
  m.def("regularized_gamma_q_backward(Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor)");
  m.def("regularized_gamma_q_backward_backward(Tensor grad_grad_a, Tensor grad_grad_x, Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor, Tensor)");

  // Hypergeometric function
  m.def("hypergeometric_2_f_1(Tensor a, Tensor b, Tensor c, Tensor z) -> Tensor");
  m.def("hypergeometric_2_f_1_backward(Tensor grad_output, Tensor a, Tensor b, Tensor c, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("hypergeometric_2_f_1_backward_backward(Tensor gg_a, Tensor gg_b, Tensor gg_c, Tensor gg_z, Tensor grad_output, Tensor a, Tensor b, Tensor c, Tensor z) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Chebyshev polynomial (special function version)
  m.def("chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor");
  m.def("chebyshev_polynomial_t_backward(Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_t_backward_backward(Tensor gg_x, Tensor gg_n, Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor, Tensor)");

  // Modified Bessel functions of the first kind
  m.def("modified_bessel_i_0(Tensor z) -> Tensor");
  m.def("modified_bessel_i_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("modified_bessel_i_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("modified_bessel_i_1(Tensor z) -> Tensor");
  m.def("modified_bessel_i_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("modified_bessel_i_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("modified_bessel_i(Tensor n, Tensor z) -> Tensor");
  m.def("modified_bessel_i_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("modified_bessel_i_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Bessel functions of the first kind
  m.def("bessel_j_0(Tensor z) -> Tensor");
  m.def("bessel_j_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("bessel_j_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("bessel_j_1(Tensor z) -> Tensor");
  m.def("bessel_j_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("bessel_j_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("bessel_j(Tensor n, Tensor z) -> Tensor");
  m.def("bessel_j_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("bessel_j_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Bessel functions of the second kind
  m.def("bessel_y_0(Tensor z) -> Tensor");
  m.def("bessel_y_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("bessel_y_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("bessel_y_1(Tensor z) -> Tensor");
  m.def("bessel_y_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("bessel_y_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("bessel_y(Tensor n, Tensor z) -> Tensor");
  m.def("bessel_y_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("bessel_y_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Modified Bessel functions of the second kind
  m.def("modified_bessel_k_0(Tensor z) -> Tensor");
  m.def("modified_bessel_k_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("modified_bessel_k_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("modified_bessel_k_1(Tensor z) -> Tensor");
  m.def("modified_bessel_k_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("modified_bessel_k_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("modified_bessel_k(Tensor n, Tensor z) -> Tensor");
  m.def("modified_bessel_k_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("modified_bessel_k_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Spherical Bessel functions of the first kind
  m.def("spherical_bessel_j_0(Tensor z) -> Tensor");
  m.def("spherical_bessel_j_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("spherical_bessel_j_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("spherical_bessel_j_1(Tensor z) -> Tensor");
  m.def("spherical_bessel_j_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("spherical_bessel_j_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("spherical_bessel_j(Tensor n, Tensor z) -> Tensor");
  m.def("spherical_bessel_j_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("spherical_bessel_j_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Spherical Bessel functions of the second kind
  m.def("spherical_bessel_y_0(Tensor z) -> Tensor");
  m.def("spherical_bessel_y_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("spherical_bessel_y_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("spherical_bessel_y_1(Tensor z) -> Tensor");
  m.def("spherical_bessel_y_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("spherical_bessel_y_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("spherical_bessel_y(Tensor n, Tensor z) -> Tensor");
  m.def("spherical_bessel_y_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("spherical_bessel_y_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Modified spherical Bessel functions of the first kind
  m.def("spherical_bessel_i_0(Tensor z) -> Tensor");
  m.def("spherical_bessel_i_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("spherical_bessel_i_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("spherical_bessel_i_1(Tensor z) -> Tensor");
  m.def("spherical_bessel_i_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("spherical_bessel_i_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("spherical_bessel_i(Tensor n, Tensor z) -> Tensor");
  m.def("spherical_bessel_i_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("spherical_bessel_i_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Modified spherical Bessel functions of the second kind
  m.def("spherical_bessel_k_0(Tensor z) -> Tensor");
  m.def("spherical_bessel_k_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("spherical_bessel_k_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("spherical_bessel_k_1(Tensor z) -> Tensor");
  m.def("spherical_bessel_k_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("spherical_bessel_k_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  m.def("spherical_bessel_k(Tensor n, Tensor z) -> Tensor");
  m.def("spherical_bessel_k_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("spherical_bessel_k_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Carlson elliptic integrals
  m.def("carlson_elliptic_integral_r_f(Tensor x, Tensor y, Tensor z) -> Tensor");
  m.def("carlson_elliptic_integral_r_f_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("carlson_elliptic_integral_r_f_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  m.def("carlson_elliptic_integral_r_d(Tensor x, Tensor y, Tensor z) -> Tensor");
  m.def("carlson_elliptic_integral_r_d_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("carlson_elliptic_integral_r_d_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  m.def("carlson_elliptic_integral_r_c(Tensor x, Tensor y) -> Tensor");
  m.def("carlson_elliptic_integral_r_c_backward(Tensor grad_output, Tensor x, Tensor y) -> (Tensor, Tensor)");
  m.def("carlson_elliptic_integral_r_c_backward_backward(Tensor gg_x, Tensor gg_y, Tensor grad_output, Tensor x, Tensor y) -> (Tensor, Tensor, Tensor)");

  m.def("carlson_elliptic_integral_r_j(Tensor x, Tensor y, Tensor z, Tensor p) -> Tensor");
  m.def("carlson_elliptic_integral_r_j_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z, Tensor p) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("carlson_elliptic_integral_r_j_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor gg_p, Tensor grad_output, Tensor x, Tensor y, Tensor z, Tensor p) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.def("carlson_elliptic_integral_r_g(Tensor x, Tensor y, Tensor z) -> Tensor");
  m.def("carlson_elliptic_integral_r_g_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("carlson_elliptic_integral_r_g_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  m.def("carlson_elliptic_integral_r_e(Tensor x, Tensor y, Tensor z) -> Tensor");
  m.def("carlson_elliptic_integral_r_e_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("carlson_elliptic_integral_r_e_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  m.def("carlson_elliptic_integral_r_m(Tensor x, Tensor y, Tensor z) -> Tensor");
  m.def("carlson_elliptic_integral_r_m_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("carlson_elliptic_integral_r_m_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  m.def("carlson_elliptic_integral_r_k(Tensor x, Tensor y) -> Tensor");
  m.def("carlson_elliptic_integral_r_k_backward(Tensor grad_output, Tensor x, Tensor y) -> (Tensor, Tensor)");
  m.def("carlson_elliptic_integral_r_k_backward_backward(Tensor gg_x, Tensor gg_y, Tensor grad_output, Tensor x, Tensor y) -> (Tensor, Tensor, Tensor)");

  // Legendre elliptic integrals
  m.def("complete_legendre_elliptic_integral_k(Tensor m) -> Tensor");
  m.def("complete_legendre_elliptic_integral_k_backward(Tensor grad_output, Tensor m) -> Tensor");
  m.def("complete_legendre_elliptic_integral_k_backward_backward(Tensor gg_m, Tensor grad_output, Tensor m) -> (Tensor, Tensor)");

  m.def("complete_legendre_elliptic_integral_e(Tensor m) -> Tensor");
  m.def("complete_legendre_elliptic_integral_e_backward(Tensor grad_output, Tensor m) -> Tensor");
  m.def("complete_legendre_elliptic_integral_e_backward_backward(Tensor gg_m, Tensor grad_output, Tensor m) -> (Tensor, Tensor)");

  m.def("incomplete_legendre_elliptic_integral_e(Tensor phi, Tensor m) -> Tensor");
  m.def("incomplete_legendre_elliptic_integral_e_backward(Tensor grad_output, Tensor phi, Tensor m) -> (Tensor, Tensor)");
  m.def("incomplete_legendre_elliptic_integral_e_backward_backward(Tensor gg_phi, Tensor gg_m, Tensor grad_output, Tensor phi, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("incomplete_legendre_elliptic_integral_f(Tensor phi, Tensor m) -> Tensor");
  m.def("incomplete_legendre_elliptic_integral_f_backward(Tensor grad_output, Tensor phi, Tensor m) -> (Tensor, Tensor)");
  m.def("incomplete_legendre_elliptic_integral_f_backward_backward(Tensor gg_phi, Tensor gg_m, Tensor grad_output, Tensor phi, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("complete_legendre_elliptic_integral_pi(Tensor n, Tensor m) -> Tensor");
  m.def("complete_legendre_elliptic_integral_pi_backward(Tensor grad_output, Tensor n, Tensor m) -> (Tensor, Tensor)");
  m.def("complete_legendre_elliptic_integral_pi_backward_backward(Tensor gg_n, Tensor gg_m, Tensor grad_output, Tensor n, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("incomplete_legendre_elliptic_integral_pi(Tensor n, Tensor phi, Tensor m) -> Tensor");
  m.def("incomplete_legendre_elliptic_integral_pi_backward(Tensor grad_output, Tensor n, Tensor phi, Tensor m) -> (Tensor, Tensor, Tensor)");
  m.def("incomplete_legendre_elliptic_integral_pi_backward_backward(Tensor gg_n, Tensor gg_phi, Tensor gg_m, Tensor grad_output, Tensor n, Tensor phi, Tensor m) -> (Tensor, Tensor, Tensor, Tensor)");

  // Jacobi elliptic functions
  m.def("jacobi_amplitude_am(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_amplitude_am_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_amplitude_am_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_dn(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_dn_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_dn_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_cn(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_cn_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_cn_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_sn(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_sn_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_sn_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_sd(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_sd_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_sd_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_cd(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_cd_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_cd_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_sc(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_sc_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_sc_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_nd(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_nd_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_nd_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_nc(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_nc_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_nc_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_ns(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_ns_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_ns_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_dc(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_dc_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_dc_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_ds(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_ds_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_ds_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("jacobi_elliptic_cs(Tensor u, Tensor m) -> Tensor");
  m.def("jacobi_elliptic_cs_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  m.def("jacobi_elliptic_cs_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  // Inverse Jacobi elliptic functions
  m.def("inverse_jacobi_elliptic_sn(Tensor x, Tensor m) -> Tensor");
  m.def("inverse_jacobi_elliptic_sn_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  m.def("inverse_jacobi_elliptic_sn_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("inverse_jacobi_elliptic_cn(Tensor x, Tensor m) -> Tensor");
  m.def("inverse_jacobi_elliptic_cn_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  m.def("inverse_jacobi_elliptic_cn_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("inverse_jacobi_elliptic_dn(Tensor x, Tensor m) -> Tensor");
  m.def("inverse_jacobi_elliptic_dn_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  m.def("inverse_jacobi_elliptic_dn_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("inverse_jacobi_elliptic_sd(Tensor x, Tensor m) -> Tensor");
  m.def("inverse_jacobi_elliptic_sd_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  m.def("inverse_jacobi_elliptic_sd_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("inverse_jacobi_elliptic_cd(Tensor x, Tensor m) -> Tensor");
  m.def("inverse_jacobi_elliptic_cd_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  m.def("inverse_jacobi_elliptic_cd_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  m.def("inverse_jacobi_elliptic_sc(Tensor x, Tensor m) -> Tensor");
  m.def("inverse_jacobi_elliptic_sc_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  m.def("inverse_jacobi_elliptic_sc_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  // Jacobi theta functions
  m.def("theta_1(Tensor z, Tensor q) -> Tensor");
  m.def("theta_1_backward(Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor)");
  m.def("theta_1_backward_backward(Tensor gg_z, Tensor gg_q, Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor, Tensor)");

  m.def("theta_2(Tensor z, Tensor q) -> Tensor");
  m.def("theta_2_backward(Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor)");
  m.def("theta_2_backward_backward(Tensor gg_z, Tensor gg_q, Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor, Tensor)");

  m.def("theta_3(Tensor z, Tensor q) -> Tensor");
  m.def("theta_3_backward(Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor)");
  m.def("theta_3_backward_backward(Tensor gg_z, Tensor gg_q, Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor, Tensor)");

  m.def("theta_4(Tensor z, Tensor q) -> Tensor");
  m.def("theta_4_backward(Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor)");
  m.def("theta_4_backward_backward(Tensor gg_z, Tensor gg_q, Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor, Tensor)");

  // Exponential integrals
  m.def("exponential_integral_ei(Tensor x) -> Tensor");
  m.def("exponential_integral_ei_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("exponential_integral_ei_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  m.def("exponential_integral_e_1(Tensor x) -> Tensor");
  m.def("exponential_integral_e_1_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("exponential_integral_e_1_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  m.def("exponential_integral_ein(Tensor x) -> Tensor");
  m.def("exponential_integral_ein_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("exponential_integral_ein_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  m.def("exponential_integral_e(Tensor n, Tensor x) -> Tensor");
  m.def("exponential_integral_e_backward(Tensor grad_output, Tensor n, Tensor x) -> (Tensor, Tensor)");
  m.def("exponential_integral_e_backward_backward(Tensor gg_n, Tensor gg_x, Tensor grad_output, Tensor n, Tensor x) -> (Tensor, Tensor, Tensor)");

  // Sine integral
  m.def("sine_integral_si(Tensor x) -> Tensor");
  m.def("sine_integral_si_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("sine_integral_si_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  // Cosine integral
  m.def("cosine_integral_ci(Tensor x) -> Tensor");
  m.def("cosine_integral_ci_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("cosine_integral_ci_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  // Spherical Hankel functions of the first kind
  m.def("spherical_hankel_1(Tensor n, Tensor z) -> Tensor");
  m.def("spherical_hankel_1_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("spherical_hankel_1_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Spherical Hankel functions of the second kind
  m.def("spherical_hankel_2(Tensor n, Tensor z) -> Tensor");
  m.def("spherical_hankel_2_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("spherical_hankel_2_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");
}

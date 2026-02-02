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

  // Reciprocal gamma function
  m.def("reciprocal_gamma(Tensor z) -> Tensor");
  m.def("reciprocal_gamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("reciprocal_gamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Gamma sign function
  m.def("gamma_sign(Tensor x) -> Tensor");
  m.def("gamma_sign_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("gamma_sign_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

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

  // Confluent hypergeometric function M (Kummer's function of the first kind)
  m.def("confluent_hypergeometric_m(Tensor a, Tensor b, Tensor z) -> Tensor");
  m.def("confluent_hypergeometric_m_backward(Tensor grad_output, Tensor a, Tensor b, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("confluent_hypergeometric_m_backward_backward(Tensor gg_a, Tensor gg_b, Tensor gg_z, Tensor grad_output, Tensor a, Tensor b, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  // Confluent hypergeometric function U (Kummer's function of the second kind)
  m.def("confluent_hypergeometric_u(Tensor a, Tensor b, Tensor z) -> Tensor");
  m.def("confluent_hypergeometric_u_backward(Tensor grad, Tensor a, Tensor b, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("confluent_hypergeometric_u_backward_backward(Tensor gg_a, Tensor gg_b, Tensor gg_z, Tensor grad, Tensor a, Tensor b, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  // Whittaker M function
  m.def("whittaker_m(Tensor kappa, Tensor mu, Tensor z) -> Tensor");
  m.def("whittaker_m_backward(Tensor grad, Tensor kappa, Tensor mu, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("whittaker_m_backward_backward(Tensor gg_kappa, Tensor gg_mu, Tensor gg_z, Tensor grad, Tensor kappa, Tensor mu, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  // Whittaker W function
  m.def("whittaker_w(Tensor kappa, Tensor mu, Tensor z) -> Tensor");
  m.def("whittaker_w_backward(Tensor grad, Tensor kappa, Tensor mu, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("whittaker_w_backward_backward(Tensor gg_kappa, Tensor gg_mu, Tensor gg_z, Tensor grad, Tensor kappa, Tensor mu, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  // Hypergeometric 0F1 (confluent hypergeometric limit function)
  m.def("hypergeometric_0_f_1(Tensor b, Tensor z) -> Tensor");
  m.def("hypergeometric_0_f_1_backward(Tensor grad, Tensor b, Tensor z) -> (Tensor, Tensor)");
  m.def("hypergeometric_0_f_1_backward_backward(Tensor gg_b, Tensor gg_z, Tensor grad, Tensor b, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Hypergeometric 1F2
  m.def("hypergeometric_1_f_2(Tensor a, Tensor b1, Tensor b2, Tensor z) -> Tensor");
  m.def("hypergeometric_1_f_2_backward(Tensor grad, Tensor a, Tensor b1, Tensor b2, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("hypergeometric_1_f_2_backward_backward(Tensor gg_a, Tensor gg_b1, Tensor gg_b2, Tensor gg_z, Tensor grad, Tensor a, Tensor b1, Tensor b2, Tensor z) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Generalized hypergeometric pFq
  // a has shape [..., p], b has shape [..., q], z has shape [...], output has shape [...]
  m.def("hypergeometric_p_f_q(Tensor a, Tensor b, Tensor z) -> Tensor");
  m.def("hypergeometric_p_f_q_backward(Tensor grad, Tensor a, Tensor b, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("hypergeometric_p_f_q_backward_backward(Tensor gg_a, Tensor gg_b, Tensor gg_z, Tensor grad, Tensor a, Tensor b, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  // Chebyshev polynomial of the first kind T_n(x)
  m.def("chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor");
  m.def("chebyshev_polynomial_t_backward(Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_t_backward_backward(Tensor gg_x, Tensor gg_n, Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor, Tensor)");

  // Chebyshev polynomial of the second kind U_n(x)
  m.def("chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor");
  m.def("chebyshev_polynomial_u_backward(Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_u_backward_backward(Tensor gg_x, Tensor gg_n, Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor, Tensor)");

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

  // Weierstrass elliptic function P
  m.def("weierstrass_p(Tensor z, Tensor g2, Tensor g3) -> Tensor");
  m.def("weierstrass_p_backward(Tensor grad_output, Tensor z, Tensor g2, Tensor g3) -> (Tensor, Tensor, Tensor)");
  m.def("weierstrass_p_backward_backward(Tensor gg_z, Tensor gg_g2, Tensor gg_g3, Tensor grad_output, Tensor z, Tensor g2, Tensor g3) -> (Tensor, Tensor, Tensor, Tensor)");

  // Weierstrass sigma function
  m.def("weierstrass_sigma(Tensor z, Tensor g2, Tensor g3) -> Tensor");
  m.def("weierstrass_sigma_backward(Tensor grad_output, Tensor z, Tensor g2, Tensor g3) -> (Tensor, Tensor, Tensor)");
  m.def("weierstrass_sigma_backward_backward(Tensor gg_z, Tensor gg_g2, Tensor gg_g3, Tensor grad_output, Tensor z, Tensor g2, Tensor g3) -> (Tensor, Tensor, Tensor, Tensor)");

  // Weierstrass zeta function
  m.def("weierstrass_zeta(Tensor z, Tensor g2, Tensor g3) -> Tensor");
  m.def("weierstrass_zeta_backward(Tensor grad_output, Tensor z, Tensor g2, Tensor g3) -> (Tensor, Tensor, Tensor)");
  m.def("weierstrass_zeta_backward_backward(Tensor gg_z, Tensor gg_g2, Tensor gg_g3, Tensor grad_output, Tensor z, Tensor g2, Tensor g3) -> (Tensor, Tensor, Tensor, Tensor)");

  // Weierstrass eta quasi-period
  m.def("weierstrass_eta(Tensor g2, Tensor g3) -> Tensor");
  m.def("weierstrass_eta_backward(Tensor grad_output, Tensor g2, Tensor g3) -> (Tensor, Tensor)");
  m.def("weierstrass_eta_backward_backward(Tensor gg_g2, Tensor gg_g3, Tensor grad_output, Tensor g2, Tensor g3) -> (Tensor, Tensor, Tensor)");

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

  // Airy function of the first kind
  m.def("airy_ai(Tensor x) -> Tensor");
  m.def("airy_ai_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("airy_ai_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  // Airy function of the second kind
  m.def("airy_bi(Tensor x) -> Tensor");
  m.def("airy_bi_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("airy_bi_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  // Lambert W function (product logarithm)
  m.def("lambert_w(Tensor k, Tensor z) -> Tensor");
  m.def("lambert_w_backward(Tensor grad_output, Tensor k, Tensor z) -> (Tensor, Tensor)");
  m.def("lambert_w_backward_backward(Tensor gg_k, Tensor gg_z, Tensor grad_output, Tensor k, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Kelvin function ber (real part of J_0 at rotated argument)
  m.def("kelvin_ber(Tensor x) -> Tensor");
  m.def("kelvin_ber_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("kelvin_ber_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  // Kelvin function bei (imaginary part of J_0 at rotated argument)
  m.def("kelvin_bei(Tensor x) -> Tensor");
  m.def("kelvin_bei_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("kelvin_bei_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  // Kelvin function ker (real part of K_0 at rotated argument)
  m.def("kelvin_ker(Tensor x) -> Tensor");
  m.def("kelvin_ker_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("kelvin_ker_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  // Kelvin function kei (imaginary part of K_0 at rotated argument)
  m.def("kelvin_kei(Tensor x) -> Tensor");
  m.def("kelvin_kei_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("kelvin_kei_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  // Riemann zeta function (s > 1 only)
  m.def("zeta(Tensor s) -> Tensor");
  m.def("zeta_backward(Tensor grad_output, Tensor s) -> Tensor");
  m.def("zeta_backward_backward(Tensor gg_s, Tensor grad_output, Tensor s) -> (Tensor, Tensor)");

  // Polylogarithm function Li_s(z)
  m.def("polylogarithm_li(Tensor s, Tensor z) -> Tensor");
  m.def("polylogarithm_li_backward(Tensor grad_output, Tensor s, Tensor z) -> (Tensor, Tensor)");
  m.def("polylogarithm_li_backward_backward(Tensor gg_s, Tensor gg_z, Tensor grad_output, Tensor s, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Parabolic cylinder functions (DLMF Chapter 12)
  m.def("parabolic_cylinder_u(Tensor a, Tensor z) -> Tensor");
  m.def("parabolic_cylinder_u_backward(Tensor grad_output, Tensor a, Tensor z) -> (Tensor, Tensor)");
  m.def("parabolic_cylinder_u_backward_backward(Tensor gg_a, Tensor gg_z, Tensor grad_output, Tensor a, Tensor z) -> (Tensor, Tensor, Tensor)");

  m.def("parabolic_cylinder_v(Tensor a, Tensor z) -> Tensor");
  m.def("parabolic_cylinder_v_backward(Tensor grad_output, Tensor a, Tensor z) -> (Tensor, Tensor)");
  m.def("parabolic_cylinder_v_backward_backward(Tensor gg_a, Tensor gg_z, Tensor grad_output, Tensor a, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Faddeeva function w(z) = exp(-z^2) * erfc(-iz)
  m.def("faddeeva_w(Tensor z) -> Tensor");
  m.def("faddeeva_w_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("faddeeva_w_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Inverse error function
  m.def("erfinv(Tensor x) -> Tensor");
  m.def("erfinv_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("erfinv_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  // Inverse complementary error function
  m.def("erfcinv(Tensor x) -> Tensor");
  m.def("erfcinv_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("erfcinv_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");

  // Fresnel sine integral
  m.def("fresnel_s(Tensor z) -> Tensor");
  m.def("fresnel_s_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("fresnel_s_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Fresnel cosine integral
  m.def("fresnel_c(Tensor z) -> Tensor");
  m.def("fresnel_c_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("fresnel_c_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Dawson's integral
  m.def("dawson(Tensor z) -> Tensor");
  m.def("dawson_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("dawson_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Voigt profile
  m.def("voigt_profile(Tensor x, Tensor sigma, Tensor gamma) -> Tensor");
  m.def("voigt_profile_backward(Tensor grad_output, Tensor x, Tensor sigma, Tensor gamma) -> (Tensor, Tensor, Tensor)");
  m.def("voigt_profile_backward_backward(Tensor gg_x, Tensor gg_sigma, Tensor gg_gamma, Tensor grad_output, Tensor x, Tensor sigma, Tensor gamma) -> (Tensor, Tensor, Tensor, Tensor)");

  // Legendre polynomial P_n(z)
  m.def("legendre_polynomial_p(Tensor n, Tensor z) -> Tensor");
  m.def("legendre_polynomial_p_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("legendre_polynomial_p_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Legendre function of the second kind Q_n(x)
  m.def("legendre_polynomial_q(Tensor x, Tensor n) -> Tensor");
  m.def("legendre_polynomial_q_backward(Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor)");
  m.def("legendre_polynomial_q_backward_backward(Tensor gg_x, Tensor gg_n, Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor, Tensor)");

  // Hermite polynomial (physicists') H_n(z)
  m.def("hermite_polynomial_h(Tensor n, Tensor z) -> Tensor");
  m.def("hermite_polynomial_h_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("hermite_polynomial_h_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Hermite polynomial (probabilists') He_n(z)
  m.def("hermite_polynomial_he(Tensor n, Tensor z) -> Tensor");
  m.def("hermite_polynomial_he_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("hermite_polynomial_he_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Generalized Laguerre polynomial L_n^alpha(z)
  m.def("laguerre_polynomial_l(Tensor n, Tensor alpha, Tensor z) -> Tensor");
  m.def("laguerre_polynomial_l_backward(Tensor grad_output, Tensor n, Tensor alpha, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("laguerre_polynomial_l_backward_backward(Tensor gg_n, Tensor gg_alpha, Tensor gg_z, Tensor grad_output, Tensor n, Tensor alpha, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  // Gegenbauer (ultraspherical) polynomial C_n^lambda(z)
  m.def("gegenbauer_polynomial_c(Tensor n, Tensor lambda, Tensor z) -> Tensor");
  m.def("gegenbauer_polynomial_c_backward(Tensor grad_output, Tensor n, Tensor lambda, Tensor z) -> (Tensor, Tensor, Tensor)");
  m.def("gegenbauer_polynomial_c_backward_backward(Tensor gg_n, Tensor gg_lambda, Tensor gg_z, Tensor grad_output, Tensor n, Tensor lambda, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  // Jacobi polynomial P_n^(alpha,beta)(z)
  m.def("jacobi_polynomial_p(Tensor n, Tensor alpha, Tensor beta, Tensor z) -> Tensor");
  m.def("jacobi_polynomial_p_backward(Tensor grad_output, Tensor n, Tensor alpha, Tensor beta, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("jacobi_polynomial_p_backward_backward(Tensor gg_n, Tensor gg_alpha, Tensor gg_beta, Tensor gg_z, Tensor grad_output, Tensor n, Tensor alpha, Tensor beta, Tensor z) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Radial Zernike polynomial R_n^m(rho)
  m.def("zernike_polynomial_r(Tensor n, Tensor m, Tensor rho) -> Tensor");
  m.def("zernike_polynomial_r_backward(Tensor grad_output, Tensor n, Tensor m, Tensor rho) -> (Tensor, Tensor, Tensor)");
  m.def("zernike_polynomial_r_backward_backward(Tensor gg_n, Tensor gg_m, Tensor gg_rho, Tensor grad_output, Tensor n, Tensor m, Tensor rho) -> (Tensor, Tensor, Tensor, Tensor)");

  // Full Zernike polynomial Z_n^m(rho, theta)
  m.def("zernike_polynomial_z(Tensor n, Tensor m, Tensor rho, Tensor theta) -> Tensor");
  m.def("zernike_polynomial_z_backward(Tensor grad_output, Tensor n, Tensor m, Tensor rho, Tensor theta) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("zernike_polynomial_z_backward_backward(Tensor gg_n, Tensor gg_m, Tensor gg_rho, Tensor gg_theta, Tensor grad_output, Tensor n, Tensor m, Tensor rho, Tensor theta) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Krawtchouk polynomial K_n(x; p, N)
  m.def("krawtchouk_polynomial_k(Tensor n, Tensor x, Tensor p, Tensor N) -> Tensor");
  m.def("krawtchouk_polynomial_k_backward(Tensor grad, Tensor n, Tensor x, Tensor p, Tensor N) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("krawtchouk_polynomial_k_backward_backward(Tensor gg_n, Tensor gg_x, Tensor gg_p, Tensor gg_N, Tensor grad, Tensor n, Tensor x, Tensor p, Tensor N) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Meixner polynomial M_n(x; beta, c)
  m.def("meixner_polynomial_m(Tensor n, Tensor x, Tensor beta, Tensor c) -> Tensor");
  m.def("meixner_polynomial_m_backward(Tensor grad, Tensor n, Tensor x, Tensor beta, Tensor c) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("meixner_polynomial_m_backward_backward(Tensor gg_n, Tensor gg_x, Tensor gg_beta, Tensor gg_c, Tensor grad, Tensor n, Tensor x, Tensor beta, Tensor c) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Hahn polynomial Q_n(x; alpha, beta, N)
  m.def("hahn_polynomial_q(Tensor n, Tensor x, Tensor alpha, Tensor beta, Tensor N) -> Tensor");
  m.def("hahn_polynomial_q_backward(Tensor grad, Tensor n, Tensor x, Tensor alpha, Tensor beta, Tensor N) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("hahn_polynomial_q_backward_backward(Tensor gg_n, Tensor gg_x, Tensor gg_alpha, Tensor gg_beta, Tensor gg_N, Tensor grad, Tensor n, Tensor x, Tensor alpha, Tensor beta, Tensor N) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Charlier polynomial C_n(x; a)
  m.def("charlier_polynomial_c(Tensor n, Tensor x, Tensor a) -> Tensor");
  m.def("charlier_polynomial_c_backward(Tensor grad, Tensor n, Tensor x, Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("charlier_polynomial_c_backward_backward(Tensor gg_n, Tensor gg_x, Tensor gg_a, Tensor grad, Tensor n, Tensor x, Tensor a) -> (Tensor, Tensor, Tensor, Tensor)");

  // Pochhammer symbol (rising factorial)
  m.def("pochhammer(Tensor z, Tensor m) -> Tensor");
  m.def("pochhammer_backward(Tensor grad_output, Tensor z, Tensor m) -> (Tensor, Tensor)");
  m.def("pochhammer_backward_backward(Tensor gg_z, Tensor gg_m, Tensor grad_output, Tensor z, Tensor m) -> (Tensor, Tensor, Tensor)");

  // Log multivariate gamma function
  m.def("log_multivariate_gamma(Tensor a, int d) -> Tensor");
  m.def("log_multivariate_gamma_backward(Tensor grad_output, Tensor a, int d) -> Tensor");
  m.def("log_multivariate_gamma_backward_backward(Tensor gg_a, Tensor grad_output, Tensor a, int d) -> (Tensor, Tensor)");

  // Inverse regularized gamma P function
  m.def("inverse_regularized_gamma_p(Tensor a, Tensor y) -> Tensor");
  m.def("inverse_regularized_gamma_p_backward(Tensor grad_output, Tensor a, Tensor y) -> (Tensor, Tensor)");
  m.def("inverse_regularized_gamma_p_backward_backward(Tensor gg_a, Tensor gg_y, Tensor grad_output, Tensor a, Tensor y) -> (Tensor, Tensor, Tensor)");

  // Inverse regularized gamma Q function
  m.def("inverse_regularized_gamma_q(Tensor a, Tensor y) -> Tensor");
  m.def("inverse_regularized_gamma_q_backward(Tensor grad_output, Tensor a, Tensor y) -> (Tensor, Tensor)");
  m.def("inverse_regularized_gamma_q_backward_backward(Tensor gg_a, Tensor gg_y, Tensor grad_output, Tensor a, Tensor y) -> (Tensor, Tensor, Tensor)");

  // Inverse regularized incomplete beta function
  m.def("inverse_regularized_incomplete_beta(Tensor a, Tensor b, Tensor y) -> Tensor");
  m.def("inverse_regularized_incomplete_beta_backward(Tensor grad_output, Tensor a, Tensor b, Tensor y) -> (Tensor, Tensor, Tensor)");
  m.def("inverse_regularized_incomplete_beta_backward_backward(Tensor gg_a, Tensor gg_b, Tensor gg_y, Tensor grad_output, Tensor a, Tensor b, Tensor y) -> (Tensor, Tensor, Tensor, Tensor)");

  // Inverse complementary regularized incomplete beta function
  m.def("inverse_complementary_regularized_incomplete_beta(Tensor a, Tensor b, Tensor y) -> Tensor");
  m.def("inverse_complementary_regularized_incomplete_beta_backward(Tensor grad_output, Tensor a, Tensor b, Tensor y) -> (Tensor, Tensor, Tensor)");
  m.def("inverse_complementary_regularized_incomplete_beta_backward_backward(Tensor gg_a, Tensor gg_b, Tensor gg_y, Tensor grad_output, Tensor a, Tensor b, Tensor y) -> (Tensor, Tensor, Tensor, Tensor)");
  // Struve function H_0
  m.def("struve_h_0(Tensor z) -> Tensor");
  m.def("struve_h_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("struve_h_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Struve function H_1
  m.def("struve_h_1(Tensor z) -> Tensor");
  m.def("struve_h_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("struve_h_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Modified Struve function L_0
  m.def("struve_l_0(Tensor z) -> Tensor");
  m.def("struve_l_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("struve_l_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Modified Struve function L_1
  m.def("struve_l_1(Tensor z) -> Tensor");
  m.def("struve_l_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("struve_l_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // General order Struve function H_n(z)
  m.def("struve_h(Tensor n, Tensor z) -> Tensor");
  m.def("struve_h_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("struve_h_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // General order modified Struve function L_n(z)
  m.def("struve_l(Tensor n, Tensor z) -> Tensor");
  m.def("struve_l_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("struve_l_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Anger function J_nu(z)
  m.def("anger_j(Tensor n, Tensor z) -> Tensor");
  m.def("anger_j_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("anger_j_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Weber function E_nu(z)
  m.def("weber_e(Tensor n, Tensor z) -> Tensor");
  m.def("weber_e_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  m.def("weber_e_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");
}

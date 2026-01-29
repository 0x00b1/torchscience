// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// Standard polynomial - CPU
#include "cpu/polynomial/polynomial/evaluate.h"
#include "cpu/polynomial/polynomial/derivative.h"
#include "cpu/polynomial/polynomial/antiderivative.h"
#include "cpu/polynomial/polynomial/add.h"
#include "cpu/polynomial/polynomial/subtract.h"
#include "cpu/polynomial/polynomial/negate.h"
#include "cpu/polynomial/polynomial/scale.h"
#include "cpu/polynomial/polynomial/multiply.h"
#include "cpu/polynomial/polynomial/divmod.h"

// Standard polynomial - Meta
#include "meta/polynomial/polynomial/evaluate.h"
#include "meta/polynomial/polynomial/derivative.h"
#include "meta/polynomial/polynomial/antiderivative.h"
#include "meta/polynomial/polynomial/add.h"
#include "meta/polynomial/polynomial/subtract.h"
#include "meta/polynomial/polynomial/negate.h"
#include "meta/polynomial/polynomial/scale.h"
#include "meta/polynomial/polynomial/multiply.h"
#include "meta/polynomial/polynomial/divmod.h"

// Standard polynomial - Autograd
#include "autograd/polynomial/polynomial/evaluate.h"
#include "autograd/polynomial/polynomial/polynomial_derivative.h"
#include "autograd/polynomial/polynomial/polynomial_antiderivative.h"
#include "autograd/polynomial/polynomial/polynomial_add.h"
#include "autograd/polynomial/polynomial/polynomial_subtract.h"
#include "autograd/polynomial/polynomial/polynomial_negate.h"
#include "autograd/polynomial/polynomial/polynomial_scale.h"
#include "autograd/polynomial/polynomial/polynomial_multiply.h"
#include "autograd/polynomial/polynomial/polynomial_divmod.h"

// Standard polynomial - Autocast
#include "autocast/polynomial/polynomial/polynomial_evaluate.h"
#include "autocast/polynomial/polynomial/polynomial_derivative.h"
#include "autocast/polynomial/polynomial/polynomial_antiderivative.h"
#include "autocast/polynomial/polynomial/polynomial_add.h"
#include "autocast/polynomial/polynomial/polynomial_subtract.h"
#include "autocast/polynomial/polynomial/polynomial_negate.h"
#include "autocast/polynomial/polynomial/polynomial_scale.h"
#include "autocast/polynomial/polynomial/polynomial_multiply.h"
#include "autocast/polynomial/polynomial/polynomial_divmod.h"

// Chebyshev T - all backends
#include "cpu/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_evaluate.h"
#include "cpu/polynomial/chebyshev_polynomial_t/multiply.h"
#include "cpu/polynomial/chebyshev_polynomial_t/derivative.h"
#include "cpu/polynomial/chebyshev_polynomial_t/mulx.h"
#include "cpu/polynomial/chebyshev_polynomial_t/antiderivative.h"
#include "meta/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_evaluate.h"
#include "meta/polynomial/chebyshev_polynomial_t/multiply.h"
#include "meta/polynomial/chebyshev_polynomial_t/derivative.h"
#include "meta/polynomial/chebyshev_polynomial_t/mulx.h"
#include "meta/polynomial/chebyshev_polynomial_t/antiderivative.h"
#include "autograd/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_evaluate.h"
#include "autograd/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_multiply.h"
#include "autograd/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_derivative.h"
#include "autograd/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_mulx.h"
#include "autograd/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_antiderivative.h"
#include "autocast/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_evaluate.h"
#include "autocast/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_multiply.h"
#include "autocast/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_derivative.h"
#include "autocast/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_mulx.h"
#include "autocast/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_antiderivative.h"

// Chebyshev U - all backends
#include "cpu/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_evaluate.h"
#include "cpu/polynomial/chebyshev_polynomial_u/multiply.h"
#include "cpu/polynomial/chebyshev_polynomial_u/derivative.h"
#include "cpu/polynomial/chebyshev_polynomial_u/mulx.h"
#include "cpu/polynomial/chebyshev_polynomial_u/antiderivative.h"
#include "meta/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_evaluate.h"
#include "meta/polynomial/chebyshev_polynomial_u/multiply.h"
#include "meta/polynomial/chebyshev_polynomial_u/derivative.h"
#include "meta/polynomial/chebyshev_polynomial_u/mulx.h"
#include "meta/polynomial/chebyshev_polynomial_u/antiderivative.h"
#include "autograd/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_evaluate.h"
#include "autograd/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_multiply.h"
#include "autograd/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_derivative.h"
#include "autograd/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_mulx.h"
#include "autograd/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_antiderivative.h"
#include "autocast/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_evaluate.h"
#include "autocast/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_multiply.h"
#include "autocast/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_derivative.h"
#include "autocast/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_mulx.h"
#include "autocast/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_antiderivative.h"

// Chebyshev V - all backends
#include "cpu/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_evaluate.h"
#include "cpu/polynomial/chebyshev_polynomial_v/multiply.h"
#include "cpu/polynomial/chebyshev_polynomial_v/derivative.h"
#include "cpu/polynomial/chebyshev_polynomial_v/mulx.h"
#include "cpu/polynomial/chebyshev_polynomial_v/antiderivative.h"
#include "meta/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_evaluate.h"
#include "meta/polynomial/chebyshev_polynomial_v/multiply.h"
#include "meta/polynomial/chebyshev_polynomial_v/derivative.h"
#include "meta/polynomial/chebyshev_polynomial_v/mulx.h"
#include "meta/polynomial/chebyshev_polynomial_v/antiderivative.h"
#include "autograd/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_evaluate.h"
#include "autograd/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_multiply.h"
#include "autograd/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_derivative.h"
#include "autograd/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_mulx.h"
#include "autograd/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_antiderivative.h"
#include "autocast/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_evaluate.h"
#include "autocast/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_multiply.h"
#include "autocast/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_derivative.h"
#include "autocast/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_mulx.h"
#include "autocast/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_antiderivative.h"

// Chebyshev W - all backends
#include "cpu/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_evaluate.h"
#include "cpu/polynomial/chebyshev_polynomial_w/multiply.h"
#include "cpu/polynomial/chebyshev_polynomial_w/derivative.h"
#include "cpu/polynomial/chebyshev_polynomial_w/mulx.h"
#include "cpu/polynomial/chebyshev_polynomial_w/antiderivative.h"
#include "meta/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_evaluate.h"
#include "meta/polynomial/chebyshev_polynomial_w/multiply.h"
#include "meta/polynomial/chebyshev_polynomial_w/derivative.h"
#include "meta/polynomial/chebyshev_polynomial_w/mulx.h"
#include "meta/polynomial/chebyshev_polynomial_w/antiderivative.h"
#include "autograd/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_evaluate.h"
#include "autograd/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_multiply.h"
#include "autograd/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_derivative.h"
#include "autograd/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_mulx.h"
#include "autograd/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_antiderivative.h"
#include "autocast/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_evaluate.h"
#include "autocast/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_multiply.h"
#include "autocast/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_derivative.h"
#include "autocast/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_mulx.h"
#include "autocast/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_antiderivative.h"

// Legendre P - all backends
#include "cpu/polynomial/legendre_polynomial_p/legendre_polynomial_p_evaluate.h"
#include "cpu/polynomial/legendre_polynomial_p/derivative.h"
#include "cpu/polynomial/legendre_polynomial_p/antiderivative.h"
#include "cpu/polynomial/legendre_polynomial_p/mulx.h"
#include "cpu/polynomial/legendre_polynomial_p/multiply.h"
#include "meta/polynomial/legendre_polynomial_p/legendre_polynomial_p_evaluate.h"
#include "meta/polynomial/legendre_polynomial_p/derivative.h"
#include "meta/polynomial/legendre_polynomial_p/antiderivative.h"
#include "meta/polynomial/legendre_polynomial_p/mulx.h"
#include "meta/polynomial/legendre_polynomial_p/multiply.h"
#include "autograd/polynomial/legendre_polynomial_p/legendre_polynomial_p_evaluate.h"
#include "autograd/polynomial/legendre_polynomial_p/legendre_polynomial_p_derivative.h"
#include "autograd/polynomial/legendre_polynomial_p/legendre_polynomial_p_antiderivative.h"
#include "autograd/polynomial/legendre_polynomial_p/legendre_polynomial_p_mulx.h"
#include "autograd/polynomial/legendre_polynomial_p/legendre_polynomial_p_multiply.h"
#include "autocast/polynomial/legendre_polynomial_p/legendre_polynomial_p_evaluate.h"
#include "autocast/polynomial/legendre_polynomial_p/legendre_polynomial_p_derivative.h"
#include "autocast/polynomial/legendre_polynomial_p/legendre_polynomial_p_antiderivative.h"
#include "autocast/polynomial/legendre_polynomial_p/legendre_polynomial_p_mulx.h"
#include "autocast/polynomial/legendre_polynomial_p/legendre_polynomial_p_multiply.h"

// Laguerre L - all backends
#include "cpu/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_evaluate.h"
#include "cpu/polynomial/laguerre_polynomial_l/derivative.h"
#include "cpu/polynomial/laguerre_polynomial_l/antiderivative.h"
#include "cpu/polynomial/laguerre_polynomial_l/mulx.h"
#include "cpu/polynomial/laguerre_polynomial_l/multiply.h"
#include "meta/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_evaluate.h"
#include "meta/polynomial/laguerre_polynomial_l/derivative.h"
#include "meta/polynomial/laguerre_polynomial_l/antiderivative.h"
#include "meta/polynomial/laguerre_polynomial_l/mulx.h"
#include "meta/polynomial/laguerre_polynomial_l/multiply.h"
#include "autograd/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_evaluate.h"
#include "autograd/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_derivative.h"
#include "autograd/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_antiderivative.h"
#include "autograd/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_mulx.h"
#include "autograd/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_multiply.h"
#include "autocast/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_evaluate.h"
#include "autocast/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_derivative.h"
#include "autocast/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_antiderivative.h"
#include "autocast/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_mulx.h"
#include "autocast/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_multiply.h"

// Hermite H (physicists') - all backends
#include "cpu/polynomial/hermite_polynomial_h/hermite_polynomial_h_evaluate.h"
#include "cpu/polynomial/hermite_polynomial_h/derivative.h"
#include "cpu/polynomial/hermite_polynomial_h/antiderivative.h"
#include "cpu/polynomial/hermite_polynomial_h/mulx.h"
#include "meta/polynomial/hermite_polynomial_h/hermite_polynomial_h_evaluate.h"
#include "meta/polynomial/hermite_polynomial_h/derivative.h"
#include "meta/polynomial/hermite_polynomial_h/antiderivative.h"
#include "meta/polynomial/hermite_polynomial_h/mulx.h"
#include "autograd/polynomial/hermite_polynomial_h/hermite_polynomial_h_evaluate.h"
#include "autograd/polynomial/hermite_polynomial_h/hermite_polynomial_h_derivative.h"
#include "autograd/polynomial/hermite_polynomial_h/hermite_polynomial_h_antiderivative.h"
#include "autograd/polynomial/hermite_polynomial_h/hermite_polynomial_h_mulx.h"
#include "autocast/polynomial/hermite_polynomial_h/hermite_polynomial_h_evaluate.h"
#include "autocast/polynomial/hermite_polynomial_h/hermite_polynomial_h_derivative.h"
#include "autocast/polynomial/hermite_polynomial_h/hermite_polynomial_h_antiderivative.h"
#include "autocast/polynomial/hermite_polynomial_h/hermite_polynomial_h_mulx.h"

// Hermite He (probabilists') - all backends
#include "cpu/polynomial/hermite_polynomial_he/hermite_polynomial_he_evaluate.h"
#include "cpu/polynomial/hermite_polynomial_he/derivative.h"
#include "cpu/polynomial/hermite_polynomial_he/antiderivative.h"
#include "cpu/polynomial/hermite_polynomial_he/mulx.h"
#include "meta/polynomial/hermite_polynomial_he/hermite_polynomial_he_evaluate.h"
#include "meta/polynomial/hermite_polynomial_he/derivative.h"
#include "meta/polynomial/hermite_polynomial_he/antiderivative.h"
#include "meta/polynomial/hermite_polynomial_he/mulx.h"
#include "autograd/polynomial/hermite_polynomial_he/hermite_polynomial_he_evaluate.h"
#include "autograd/polynomial/hermite_polynomial_he/hermite_polynomial_he_derivative.h"
#include "autograd/polynomial/hermite_polynomial_he/hermite_polynomial_he_antiderivative.h"
#include "autograd/polynomial/hermite_polynomial_he/hermite_polynomial_he_mulx.h"
#include "autocast/polynomial/hermite_polynomial_he/hermite_polynomial_he_evaluate.h"
#include "autocast/polynomial/hermite_polynomial_he/hermite_polynomial_he_derivative.h"
#include "autocast/polynomial/hermite_polynomial_he/hermite_polynomial_he_antiderivative.h"
#include "autocast/polynomial/hermite_polynomial_he/hermite_polynomial_he_mulx.h"

// Gegenbauer C (ultraspherical) - all backends
#include "cpu/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_evaluate.h"
#include "cpu/polynomial/gegenbauer_polynomial_c/mulx.h"
#include "meta/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_evaluate.h"
#include "meta/polynomial/gegenbauer_polynomial_c/mulx.h"
#include "autograd/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_evaluate.h"
#include "autograd/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_mulx.h"
#include "autocast/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_evaluate.h"
#include "autocast/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_mulx.h"

// Jacobi P - all backends
#include "cpu/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_evaluate.h"
#include "cpu/polynomial/jacobi_polynomial_p/mulx.h"
#include "meta/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_evaluate.h"
#include "meta/polynomial/jacobi_polynomial_p/mulx.h"
#include "autograd/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_evaluate.h"
#include "autograd/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_mulx.h"
#include "autocast/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_evaluate.h"
#include "autocast/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_mulx.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Standard polynomial operations
  m.def("polynomial_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  m.def("polynomial_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  m.def("polynomial_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  m.def("polynomial_derivative(Tensor coeffs) -> Tensor");
  m.def("polynomial_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("polynomial_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("polynomial_antiderivative(Tensor coeffs, Tensor constant) -> Tensor");
  m.def("polynomial_antiderivative_backward(Tensor grad_output, Tensor coeffs, Tensor constant) -> (Tensor, Tensor)");
  m.def("polynomial_antiderivative_backward_backward(Tensor gg_coeffs, Tensor gg_constant, Tensor coeffs) -> Tensor");

  m.def("polynomial_add(Tensor p, Tensor q) -> Tensor");
  m.def("polynomial_add_backward(Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor)");
  m.def("polynomial_add_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor, Tensor)");

  m.def("polynomial_subtract(Tensor p, Tensor q) -> Tensor");
  m.def("polynomial_subtract_backward(Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor)");
  m.def("polynomial_subtract_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor, Tensor)");

  m.def("polynomial_negate(Tensor p) -> Tensor");
  m.def("polynomial_negate_backward(Tensor grad_output, Tensor p) -> Tensor");
  m.def("polynomial_negate_backward_backward(Tensor gg_p, Tensor grad_output, Tensor p) -> (Tensor, Tensor)");

  m.def("polynomial_scale(Tensor p, Tensor c) -> Tensor");
  m.def("polynomial_scale_backward(Tensor grad_output, Tensor p, Tensor c) -> (Tensor, Tensor)");
  m.def("polynomial_scale_backward_backward(Tensor gg_p, Tensor gg_c, Tensor grad_output, Tensor p, Tensor c) -> (Tensor, Tensor, Tensor)");

  m.def("polynomial_multiply(Tensor p, Tensor q) -> Tensor");
  m.def("polynomial_multiply_backward(Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor)");
  m.def("polynomial_multiply_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor, Tensor)");

  m.def("polynomial_divmod(Tensor p, Tensor q) -> (Tensor, Tensor)");
  m.def("polynomial_divmod_backward(Tensor grad_Q, Tensor grad_R, Tensor Q, Tensor p, Tensor q) -> (Tensor, Tensor)");
  m.def("polynomial_divmod_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_Q, Tensor grad_R, Tensor Q, Tensor p, Tensor q) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Chebyshev T polynomial
  m.def("chebyshev_polynomial_t_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  m.def("chebyshev_polynomial_t_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_t_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  m.def("chebyshev_polynomial_t_multiply(Tensor a, Tensor b) -> Tensor");
  m.def("chebyshev_polynomial_t_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_t_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  m.def("chebyshev_polynomial_t_derivative(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_t_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_t_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("chebyshev_polynomial_t_mulx(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_t_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_t_mulx_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> (Tensor, Tensor)");

  m.def("chebyshev_polynomial_t_antiderivative(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_t_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_t_antiderivative_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> Tensor");

  // Chebyshev U polynomial
  m.def("chebyshev_polynomial_u_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  m.def("chebyshev_polynomial_u_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_u_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  m.def("chebyshev_polynomial_u_multiply(Tensor a, Tensor b) -> Tensor");
  m.def("chebyshev_polynomial_u_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_u_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  m.def("chebyshev_polynomial_u_derivative(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_u_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_u_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("chebyshev_polynomial_u_mulx(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_u_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_u_mulx_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> (Tensor, Tensor)");

  m.def("chebyshev_polynomial_u_antiderivative(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_u_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_u_antiderivative_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> Tensor");

  // Chebyshev V polynomial
  m.def("chebyshev_polynomial_v_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  m.def("chebyshev_polynomial_v_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_v_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  m.def("chebyshev_polynomial_v_multiply(Tensor a, Tensor b) -> Tensor");
  m.def("chebyshev_polynomial_v_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_v_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  m.def("chebyshev_polynomial_v_derivative(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_v_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_v_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("chebyshev_polynomial_v_mulx(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_v_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_v_mulx_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> (Tensor, Tensor)");

  m.def("chebyshev_polynomial_v_antiderivative(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_v_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_v_antiderivative_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> Tensor");

  // Chebyshev W polynomial
  m.def("chebyshev_polynomial_w_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  m.def("chebyshev_polynomial_w_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_w_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  m.def("chebyshev_polynomial_w_multiply(Tensor a, Tensor b) -> Tensor");
  m.def("chebyshev_polynomial_w_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_w_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  m.def("chebyshev_polynomial_w_derivative(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_w_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_w_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("chebyshev_polynomial_w_mulx(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_w_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_w_mulx_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> (Tensor, Tensor)");

  m.def("chebyshev_polynomial_w_antiderivative(Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_w_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("chebyshev_polynomial_w_antiderivative_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> Tensor");

  // Legendre P polynomial
  m.def("legendre_polynomial_p_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  m.def("legendre_polynomial_p_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  m.def("legendre_polynomial_p_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  m.def("legendre_polynomial_p_derivative(Tensor coeffs) -> Tensor");
  m.def("legendre_polynomial_p_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("legendre_polynomial_p_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("legendre_polynomial_p_antiderivative(Tensor coeffs) -> Tensor");
  m.def("legendre_polynomial_p_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("legendre_polynomial_p_antiderivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("legendre_polynomial_p_mulx(Tensor coeffs) -> Tensor");
  m.def("legendre_polynomial_p_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("legendre_polynomial_p_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("legendre_polynomial_p_multiply(Tensor a, Tensor b) -> Tensor");
  m.def("legendre_polynomial_p_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  m.def("legendre_polynomial_p_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  // Laguerre L polynomial
  m.def("laguerre_polynomial_l_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  m.def("laguerre_polynomial_l_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  m.def("laguerre_polynomial_l_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  m.def("laguerre_polynomial_l_derivative(Tensor coeffs) -> Tensor");
  m.def("laguerre_polynomial_l_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("laguerre_polynomial_l_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("laguerre_polynomial_l_antiderivative(Tensor coeffs) -> Tensor");
  m.def("laguerre_polynomial_l_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("laguerre_polynomial_l_antiderivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("laguerre_polynomial_l_mulx(Tensor coeffs) -> Tensor");
  m.def("laguerre_polynomial_l_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("laguerre_polynomial_l_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("laguerre_polynomial_l_multiply(Tensor a, Tensor b) -> Tensor");
  m.def("laguerre_polynomial_l_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  m.def("laguerre_polynomial_l_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  // Hermite H polynomial (physicists')
  m.def("hermite_polynomial_h_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  m.def("hermite_polynomial_h_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  m.def("hermite_polynomial_h_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  m.def("hermite_polynomial_h_derivative(Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_h_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_h_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("hermite_polynomial_h_antiderivative(Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_h_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_h_antiderivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("hermite_polynomial_h_mulx(Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_h_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_h_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Hermite He polynomial (probabilists')
  m.def("hermite_polynomial_he_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  m.def("hermite_polynomial_he_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  m.def("hermite_polynomial_he_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  m.def("hermite_polynomial_he_derivative(Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_he_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_he_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("hermite_polynomial_he_antiderivative(Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_he_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_he_antiderivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  m.def("hermite_polynomial_he_mulx(Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_he_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  m.def("hermite_polynomial_he_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Gegenbauer C polynomial (ultraspherical)
  m.def("gegenbauer_polynomial_c_evaluate(Tensor coeffs, Tensor x, Tensor alpha) -> Tensor");
  m.def("gegenbauer_polynomial_c_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x, Tensor alpha) -> (Tensor, Tensor, Tensor)");
  m.def("gegenbauer_polynomial_c_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor gg_alpha, Tensor grad_output, Tensor coeffs, Tensor x, Tensor alpha) -> (Tensor, Tensor, Tensor, Tensor)");

  m.def("gegenbauer_polynomial_c_mulx(Tensor coeffs, Tensor alpha) -> Tensor");
  m.def("gegenbauer_polynomial_c_mulx_backward(Tensor grad_output, Tensor coeffs, Tensor alpha) -> (Tensor, Tensor)");
  m.def("gegenbauer_polynomial_c_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs, Tensor alpha) -> Tensor");

  // Jacobi P polynomial
  m.def("jacobi_polynomial_p_evaluate(Tensor coeffs, Tensor x, Tensor alpha, Tensor beta) -> Tensor");
  m.def("jacobi_polynomial_p_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x, Tensor alpha, Tensor beta) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("jacobi_polynomial_p_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor gg_alpha, Tensor gg_beta, Tensor grad_output, Tensor coeffs, Tensor x, Tensor alpha, Tensor beta) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.def("jacobi_polynomial_p_mulx(Tensor coeffs, Tensor alpha, Tensor beta) -> Tensor");
  m.def("jacobi_polynomial_p_mulx_backward(Tensor grad_output, Tensor coeffs, Tensor alpha, Tensor beta) -> (Tensor, Tensor, Tensor)");
  m.def("jacobi_polynomial_p_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs, Tensor alpha, Tensor beta) -> Tensor");
}

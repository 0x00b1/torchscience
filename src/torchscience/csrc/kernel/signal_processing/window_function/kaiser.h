#pragma once

#include <cmath>
#include <algorithm>
#include "common.h"
#include "../../../kernel/special_functions/modified_bessel_i_0.h"
#include "../../../kernel/special_functions/modified_bessel_i_1.h"

namespace torchscience {
namespace kernel {
namespace window_function {

// Modified Bessel function of the first kind, order 0
// Uses the existing torchscience kernel implementation
template<typename scalar_t>
inline scalar_t bessel_i0(scalar_t x) {
  return special_functions::modified_bessel_i_0(x);
}

// Modified Bessel function of the first kind, order 1
template<typename scalar_t>
inline scalar_t bessel_i1(scalar_t x) {
  return special_functions::modified_bessel_i_1(x);
}

// Kaiser window forward
// w[k] = I_0(beta * sqrt(1 - x^2)) / I_0(beta)
// where x = (k - center) / center
template<typename scalar_t>
inline scalar_t kaiser(int64_t i, int64_t n, scalar_t beta, bool periodic) {
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  scalar_t center = denom / scalar_t(2);
  scalar_t x = (scalar_t(i) - center) / center;

  // Clamp to avoid numerical issues at boundaries
  scalar_t one_minus_x2 = scalar_t(1) - x * x;
  if (one_minus_x2 < scalar_t(0)) {
    one_minus_x2 = scalar_t(0);
  }
  scalar_t sqrt_term = std::sqrt(one_minus_x2);
  scalar_t arg = beta * sqrt_term;

  scalar_t i0_arg = bessel_i0(arg);
  scalar_t i0_beta = bessel_i0(beta);

  // Avoid division by zero for beta = 0 (rectangular window)
  if (i0_beta == scalar_t(0)) {
    return scalar_t(1);
  }

  return i0_arg / i0_beta;
}

// Kaiser window backward (gradient w.r.t. beta)
// d/d_beta[I_0(arg)/I_0(beta)] where arg = beta * sqrt(1 - x^2)
// = (I_1(arg) * sqrt(1-x^2) * I_0(beta) - I_0(arg) * I_1(beta)) / I_0(beta)^2
// = I_1(arg) * sqrt(1-x^2) / I_0(beta) - (I_0(arg)/I_0(beta)) * (I_1(beta)/I_0(beta))
// = I_1(arg) * sqrt(1-x^2) / I_0(beta) - w * I_1(beta) / I_0(beta)
template<typename scalar_t>
inline scalar_t kaiser_backward(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  scalar_t beta,
  bool periodic,
  scalar_t forward_value
) {
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  scalar_t center = denom / scalar_t(2);
  scalar_t x = (scalar_t(i) - center) / center;

  scalar_t one_minus_x2 = scalar_t(1) - x * x;
  if (one_minus_x2 < scalar_t(0)) {
    one_minus_x2 = scalar_t(0);
  }
  scalar_t sqrt_term = std::sqrt(one_minus_x2);
  scalar_t arg = beta * sqrt_term;

  scalar_t i0_beta = bessel_i0(beta);
  scalar_t i1_beta = bessel_i1(beta);
  scalar_t i1_arg = bessel_i1(arg);

  // Handle beta = 0 case (derivative is 0 for rectangular window)
  if (i0_beta == scalar_t(0)) {
    return scalar_t(0);
  }

  // dw/d_beta = I_1(arg) * sqrt_term / I_0(beta) - forward_value * I_1(beta) / I_0(beta)
  scalar_t dw_dbeta = i1_arg * sqrt_term / i0_beta - forward_value * i1_beta / i0_beta;

  return grad_out * dw_dbeta;
}

}  // namespace window_function
}  // namespace kernel
}  // namespace torchscience

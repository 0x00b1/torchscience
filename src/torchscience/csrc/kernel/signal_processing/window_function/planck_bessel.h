#pragma once

#include <cmath>
#include "common.h"
#include "planck_taper.h"
#include "kaiser.h"

namespace torchscience {
namespace kernel {
namespace window_function {

// Planck-Bessel window forward
// Combines Planck taper with Kaiser window: w[k] = planck_taper[k] * kaiser[k]
// Two parameters: epsilon (taper width) and beta (Kaiser shape)
template<typename scalar_t>
inline scalar_t planck_bessel(int64_t i, int64_t n, scalar_t epsilon, scalar_t beta, bool periodic) {
  scalar_t p = planck_taper<scalar_t>(i, n, epsilon, periodic);
  scalar_t k = kaiser<scalar_t>(i, n, beta, periodic);
  return p * k;
}

// Planck-Bessel window backward (gradients w.r.t. epsilon and beta)
// w = planck_taper * kaiser
// dw/d(epsilon) = d(planck_taper)/d(epsilon) * kaiser
// dw/d(beta) = planck_taper * d(kaiser)/d(beta)
template<typename scalar_t>
inline void planck_bessel_backward(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  scalar_t epsilon,
  scalar_t beta,
  bool periodic,
  scalar_t forward_value,
  scalar_t& grad_epsilon,
  scalar_t& grad_beta
) {
  // Get individual window values
  scalar_t p = planck_taper<scalar_t>(i, n, epsilon, periodic);
  scalar_t k = kaiser<scalar_t>(i, n, beta, periodic);

  // Gradient w.r.t. epsilon: d(planck_taper)/d(epsilon) * kaiser
  // Get planck_taper gradient - we pass grad_out=1 to get raw gradient
  scalar_t dp_deps = planck_taper_backward<scalar_t>(scalar_t(1), i, n, epsilon, periodic, p);
  grad_epsilon = grad_out * dp_deps * k;

  // Gradient w.r.t. beta: planck_taper * d(kaiser)/d(beta)
  // Get kaiser gradient - we pass grad_out=1 to get raw gradient
  scalar_t dk_dbeta = kaiser_backward<scalar_t>(scalar_t(1), i, n, beta, periodic, k);
  grad_beta = grad_out * p * dk_dbeta;
}

}  // namespace window_function
}  // namespace kernel
}  // namespace torchscience

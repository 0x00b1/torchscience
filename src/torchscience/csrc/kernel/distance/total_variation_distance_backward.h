#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "../information_theory/kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::distance {

/**
 * Compute gradient of total variation distance w.r.t. input probabilities.
 *
 * TV = (1/2) * sum_i |p_i - q_i|
 *
 * dTV/dp_i = (1/2) * sign(p_i - q_i)
 * dTV/dq_i = -(1/2) * sign(p_i - q_i)
 *
 * Note: At p_i = q_i, the gradient is undefined. We use subgradient = 0
 * (PyTorch convention for abs(0)).
 *
 * @param grad_output Upstream gradient (scalar)
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void total_variation_distance_backward_kernel(
    T grad_output,
    const T* p,
    const T* q,
    int64_t n,
    T* grad_p,
    T* grad_q
) {
  using torchscience::kernel::information_theory::get_eps;
  T eps = get_eps<T>();
  T coeff = grad_output / T(2);

  for (int64_t i = 0; i < n; ++i) {
    T diff = p[i] - q[i];
    T sign;
    if (diff > eps) {
      sign = T(1);
    } else if (diff < -eps) {
      sign = T(-1);
    } else {
      sign = T(0);  // Subgradient at 0
    }

    grad_p[i] = coeff * sign;
    grad_q[i] = -coeff * sign;
  }
}

}  // namespace torchscience::kernel::distance

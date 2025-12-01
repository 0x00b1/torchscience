#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "../information_theory/kullback_leibler_divergence.h"  // For get_eps<T>
#include "bhattacharyya_distance.h"  // For bhattacharyya_coefficient_kernel

namespace torchscience::kernel::distance {

/**
 * Compute backward pass for Bhattacharyya distance.
 *
 * D_B = -ln(BC) where BC = sum_i sqrt(p_i * q_i)
 *
 * Gradients:
 * dD_B/dp_j = -1/BC * d(BC)/dp_j = -1/BC * sqrt(q_j) / (2 * sqrt(p_j))
 *           = -sqrt(q_j) / (2 * BC * sqrt(p_j))
 *
 * dD_B/dq_j = -sqrt(p_j) / (2 * BC * sqrt(q_j))
 *
 * @param grad_output Upstream gradient
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 * @param n Size of distributions
 * @param bc Bhattacharyya coefficient (precomputed)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void bhattacharyya_distance_backward_kernel(
    T grad_output,
    const T* p,
    const T* q,
    T* grad_p,
    T* grad_q,
    int64_t n,
    T bc
) {
  using torchscience::kernel::information_theory::get_eps;
  T eps = get_eps<T>();

  // Common factor: -grad_output / (2 * BC)
  T factor = -grad_output / (T(2) * bc);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;

    T sqrt_p = std::sqrt(p_i);
    T sqrt_q = std::sqrt(q_i);

    // grad_p[i] = factor * sqrt(q_i) / sqrt(p_i)
    // grad_q[i] = factor * sqrt(p_i) / sqrt(q_i)
    grad_p[i] = factor * sqrt_q / sqrt_p;
    grad_q[i] = factor * sqrt_p / sqrt_q;
  }
}

}  // namespace torchscience::kernel::distance

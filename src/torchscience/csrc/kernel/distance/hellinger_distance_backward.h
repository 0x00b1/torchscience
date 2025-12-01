#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "../information_theory/kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::distance {

/**
 * Compute gradient of Hellinger distance w.r.t. input probabilities.
 *
 * H = (1/sqrt(2)) * sqrt(sum_i (sqrt(p_i) - sqrt(q_i))^2)
 *
 * Let S = sum_i (sqrt(p_i) - sqrt(q_i))^2
 * H = sqrt(S) / sqrt(2), so sqrt(S) = H * sqrt(2)
 *
 * dH/dS = 1 / (2 * sqrt(2) * sqrt(S))
 * dS/dp_j = (sqrt(p_j) - sqrt(q_j)) / sqrt(p_j)
 *
 * dH/dp_j = dH/dS * dS/dp_j = (sqrt(p_j) - sqrt(q_j)) / (2 * sqrt(2) * sqrt(S) * sqrt(p_j))
 *
 * Since sqrt(2) * sqrt(S) = sqrt(2*S) and sqrt(S) = H * sqrt(2):
 * sqrt(2*S) = sqrt(2) * H * sqrt(2) = 2H
 *
 * dH/dp_j = (sqrt(p_j) - sqrt(q_j)) / (4 * H * sqrt(p_j))
 *
 * @param grad_output Upstream gradient (scalar)
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param hellinger_value Precomputed Hellinger distance
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void hellinger_distance_backward_kernel(
    T grad_output,
    const T* p,
    const T* q,
    int64_t n,
    T hellinger_value,
    T* grad_p,
    T* grad_q
) {
  using torchscience::kernel::information_theory::get_eps;
  T eps = get_eps<T>();

  // Handle zero distance (identical distributions)
  if (hellinger_value < eps) {
    for (int64_t i = 0; i < n; ++i) {
      grad_p[i] = T(0);
      grad_q[i] = T(0);
    }
    return;
  }

  // Coefficient: grad_output / (4 * H)
  T coeff = grad_output / (T(4) * hellinger_value);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;

    T sqrt_p = std::sqrt(p_i);
    T sqrt_q = std::sqrt(q_i);
    T diff = sqrt_p - sqrt_q;

    // dH/dp_i = coeff * diff / sqrt_p
    grad_p[i] = coeff * diff / sqrt_p;

    // dH/dq_i = -coeff * diff / sqrt_q
    grad_q[i] = -coeff * diff / sqrt_q;
  }
}

}  // namespace torchscience::kernel::distance

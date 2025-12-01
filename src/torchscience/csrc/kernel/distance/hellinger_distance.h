#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "../information_theory/kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::distance {

/**
 * Compute Hellinger distance between two probability distributions.
 *
 * H(P, Q) = (1/sqrt(2)) * sqrt(sum_i (sqrt(p_i) - sqrt(q_i))^2)
 *         = sqrt(1 - BC(P, Q))
 *
 * where BC(P, Q) = sum_i sqrt(p_i * q_i) is the Bhattacharyya coefficient.
 *
 * Properties:
 * - Symmetric: H(P, Q) = H(Q, P)
 * - Bounded: 0 <= H(P, Q) <= 1
 * - Metric: satisfies triangle inequality
 *
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @return Hellinger distance
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T hellinger_distance_kernel(
    const T* p,
    const T* q,
    int64_t n
) {
  using torchscience::kernel::information_theory::get_eps;
  T eps = get_eps<T>();

  T sum_sq_diff = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;

    T sqrt_p = std::sqrt(p_i);
    T sqrt_q = std::sqrt(q_i);
    T diff = sqrt_p - sqrt_q;
    sum_sq_diff += diff * diff;
  }

  // H = (1/sqrt(2)) * sqrt(sum)
  return std::sqrt(sum_sq_diff / T(2));
}

}  // namespace torchscience::kernel::distance

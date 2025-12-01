#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

namespace torchscience::kernel::distance {

/**
 * Compute total variation distance between two probability distributions.
 *
 * TV(P, Q) = (1/2) * sum_i |p_i - q_i|
 *
 * Properties:
 * - Symmetric: TV(P, Q) = TV(Q, P)
 * - Bounded: 0 <= TV(P, Q) <= 1
 * - Metric: satisfies triangle inequality
 *
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @return Total variation distance
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T total_variation_distance_kernel(
    const T* p,
    const T* q,
    int64_t n
) {
  T sum_abs_diff = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T diff = p[i] - q[i];
    sum_abs_diff += diff > T(0) ? diff : -diff;  // std::abs
  }
  return sum_abs_diff / T(2);
}

}  // namespace torchscience::kernel::distance

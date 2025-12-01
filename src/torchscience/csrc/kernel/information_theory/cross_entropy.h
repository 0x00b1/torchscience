#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>()

namespace torchscience::kernel::information_theory {

/**
 * Compute cross-entropy between two probability vectors.
 *
 * H(P, Q) = -sum_i p_i * log(q_i)
 *
 * Note: H(P, Q) = H(P) + D_KL(P || Q)
 *
 * @param p Pointer to true/target probability distribution
 * @param q Pointer to predicted probability distribution
 * @param n Size of distributions
 * @param log_base_scale Scale factor for log base conversion (1.0 for natural log)
 * @return Cross-entropy value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T cross_entropy_kernel(
    const T* p,
    const T* q,
    int64_t n,
    T log_base_scale
) {
  T eps = get_eps<T>();
  T result = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i];
    T q_i = q[i] > eps ? q[i] : eps;

    // Only contribute if p_i > 0 (by convention, 0 * log(q) = 0)
    if (p_i > eps) {
      result -= p_i * std::log(q_i);
    }
  }

  return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory

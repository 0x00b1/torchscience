#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute Shannon entropy of a probability vector.
 *
 * H(P) = -sum_i p_i * log(p_i)
 *
 * By convention, 0 * log(0) = 0.
 *
 * @param p Pointer to probability distribution
 * @param n Size of distribution
 * @param log_base_scale Scale factor for log base conversion (1.0 for nats)
 * @return Shannon entropy value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T shannon_entropy_kernel(
    const T* p,
    int64_t n,
    T log_base_scale
) {
  T eps = get_eps<T>();
  T result = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;

    // Only contribute if p_i > 0 (by convention, 0 * log(0) = 0)
    if (p_i > eps) {
      result -= p_i * std::log(p_i);
    }
  }

  return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory

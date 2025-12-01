#pragma once

#include <cmath>
#include <limits>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute Renyi entropy of order alpha.
 *
 * H_alpha(P) = 1/(1-alpha) * log(sum_i p_i^alpha)
 *
 * Special cases:
 * - alpha -> 1: Shannon entropy (handled separately for numerical stability)
 * - alpha = 0: Hartley entropy = log(|support|)
 * - alpha = 2: Collision entropy = -log(sum_i p_i^2)
 * - alpha -> inf: Min-entropy = -log(max_i p_i)
 *
 * @param p Pointer to probability distribution
 * @param n Size of distribution
 * @param alpha Order of Renyi entropy (must be >= 0, != 1)
 * @param log_base_scale Scale factor for log base conversion (1.0 for nats)
 * @return Renyi entropy value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T renyi_entropy_kernel(
    const T* p,
    int64_t n,
    T alpha,
    T log_base_scale
) {
  T eps = get_eps<T>();

  // Handle special case: alpha = 0 (Hartley entropy)
  if (alpha < eps) {
    // Count support size (number of non-zero probabilities)
    int64_t support_size = 0;
    for (int64_t i = 0; i < n; ++i) {
      if (p[i] > eps) {
        support_size++;
      }
    }
    return std::log(static_cast<T>(support_size)) * log_base_scale;
  }

  // Handle special case: alpha -> infinity (min-entropy)
  if (alpha > T(100)) {  // Treat large alpha as infinity
    T max_p = T(0);
    for (int64_t i = 0; i < n; ++i) {
      if (p[i] > max_p) {
        max_p = p[i];
      }
    }
    return -std::log(max_p > eps ? max_p : eps) * log_base_scale;
  }

  // General case: H_alpha = 1/(1-alpha) * log(sum_i p_i^alpha)
  T sum_p_alpha = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    sum_p_alpha += std::pow(p_i, alpha);
  }

  return (T(1) / (T(1) - alpha)) * std::log(sum_p_alpha) * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory

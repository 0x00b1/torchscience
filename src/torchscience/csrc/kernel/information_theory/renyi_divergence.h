#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute Renyi divergence of order alpha between two probability distributions.
 *
 * D_alpha(P || Q) = 1/(alpha-1) * log(sum_i p_i^alpha * q_i^(1-alpha))
 *
 * Special cases:
 * - alpha -> 1: KL divergence D_KL(P || Q)
 * - alpha = 0.5: Bhattacharyya coefficient relation
 * - alpha = 2: Chi-squared divergence relation
 *
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param alpha Order of Renyi divergence (must be >= 0, != 1)
 * @param log_base_scale Scale factor for log base conversion (1.0 for nats)
 * @return Renyi divergence value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T renyi_divergence_kernel(
    const T* p,
    const T* q,
    int64_t n,
    T alpha,
    T log_base_scale
) {
  T eps = get_eps<T>();

  T sum = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    sum += std::pow(p_i, alpha) * std::pow(q_i, T(1) - alpha);
  }

  return (T(1) / (alpha - T(1))) * std::log(sum) * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory

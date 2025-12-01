#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "../information_theory/kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::distance {

/**
 * Compute Bhattacharyya distance between two probability distributions.
 *
 * D_B(P, Q) = -ln(BC(P, Q))
 * where BC(P, Q) = sum_i sqrt(p_i * q_i) is the Bhattacharyya coefficient.
 *
 * Related to Hellinger distance: H^2(P, Q) = 1 - exp(-D_B(P, Q))
 *
 * Properties:
 * - Symmetric: D_B(P, Q) = D_B(Q, P)
 * - Non-negative: D_B(P, Q) >= 0
 * - Zero for identical: D_B(P, P) = 0
 *
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @return Bhattacharyya distance
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T bhattacharyya_distance_kernel(
    const T* p,
    const T* q,
    int64_t n
) {
  using torchscience::kernel::information_theory::get_eps;
  T eps = get_eps<T>();

  T bc = T(0);  // Bhattacharyya coefficient
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    bc += std::sqrt(p_i * q_i);
  }

  // Clamp BC to avoid log(0)
  bc = bc > eps ? bc : eps;

  return -std::log(bc);
}

/**
 * Compute Bhattacharyya coefficient (helper for backward).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T bhattacharyya_coefficient_kernel(
    const T* p,
    const T* q,
    int64_t n
) {
  using torchscience::kernel::information_theory::get_eps;
  T eps = get_eps<T>();

  T bc = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    bc += std::sqrt(p_i * q_i);
  }

  return bc > eps ? bc : eps;
}

}  // namespace torchscience::kernel::distance
